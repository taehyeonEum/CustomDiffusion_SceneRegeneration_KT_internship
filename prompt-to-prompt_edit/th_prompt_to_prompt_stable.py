from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline  # Stable Diffusion 모델을 사용하기 위한 라이브러리
import torch.nn.functional as nnf
import numpy as np
import abc  # 추상 클래스를 정의하기 위한 모듈
import ptp_utils  # 사용자 정의 유틸리티 모듈 (이 코드에서 직접 정의되지 않음)
import seq_aligner  # 시퀀스 정렬을 위한 사용자 정의 모듈 (이 코드에서 직접 정의되지 않음)

# 필요한 설정값들을 정의
MY_TOKEN = '<replace with your token>'  # 사용자의 API 토큰
LOW_RESOURCE = False  # 저사양 모드 사용 여부
NUM_DIFFUSION_STEPS = 50  # 확산 단계의 수
GUIDANCE_SCALE = 7.5  # 가이던스 스케일 값
MAX_NUM_WORDS = 77  # 최대 단어 수
device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))  # 사용할 디바이스 설정
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)  # 모델 로드
tokenizer = ldm_stable.tokenizer  # 토크나이저 로드

class LocalBlend:
    # 로컬 블렌딩 연산을 정의하는 클래스
    def __call__(self, x_t, attention_store):
        # 호출시 실행되는 메소드
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]  # 어텐션 맵 선택 및 결합
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]  # 형태 변환
        maps = torch.cat(maps, dim=1)  # 맵들을 결합
        maps = (maps * self.alpha_layers).sum(-1).mean(1)  # 알파 레이어를 사용하여 가중치 적용
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))  # 맥스 풀링으로 마스크 생성
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))  # 인터폴레이션을 통한 크기 조정
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]  # 정규화
        mask = mask.gt(self.threshold)  # 임계값을 기준으로 마스크 이진화
        mask = (mask[:1] + mask[1:]).float()  # 마스크 합성
        x_t = x_t[:1] + mask * (x_t - x_t[:1])  # 마스크를 적용한 결과 반환
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        # 초기화 메소드
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)  # 알파 레이어 초기화
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)  # 단어 인덱스 추출
                alpha_layers[i, :, :, :, :, ind] = 1  # 알파 레이어 업데이트
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold  # 마스크 임계값 설정

# thum_anno
# EmptyControl과 AttentionStore의 상위 클래스. 따라서 빈 함수들이 많음. 
class AttentionControl(abc.ABC):
    # 어텐션 제어를 위한 추상 기본 클래스
    def step_callback(self, x_t):
        # 각 스텝 후

 # 호출되는 콜백 메소드
        return x_t
    
    def between_steps(self):
        # 스텝 사이에 호출되는 메소드
        return
    
    @property
    def num_uncond_att_layers(self):
        # 조건 없는 어텐션 레이어의 수 반환
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        # 어텐션을 수정하는 메소드, 구현 필요
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # 호출시 실행되는 메소드, 어텐션 수정 로직을 포함
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        # 상태 초기화 메소드
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        # 초기화 메소드
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

# thum_anno 빈껍데기인 AttentionControl을 그대로 output하는 함수. 
class EmptyControl(AttentionControl):
    # 어텐션 수정 없이 통과시키는 클래스
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        # 어텐션을 그대로 반환
        return attn
    
# thum_anno 본격적으로 attention을 저장하는 클래스인 것 같음! 
class AttentionStore(AttentionControl):
    # 어텐션 맵을 저장하는 클래스
    @staticmethod
    def get_empty_store():
        # 빈 어텐션 저장소 생성
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 어텐션 맵 저장 로직
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # 메모리 오버헤드 방지
            # thum_anno get_empty_store()의 출력값 포맷에 맞춰서 attention map을 저장하는 것 같다. 
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        # 스텝 사이에 호출되어 어텐션 맵을 종합
        # 매 스탭에 적절한 위치에서 호출되서 현재 스탭에서 저장된 atten들을 attention_store라는 조금 더 큰 딕셔너리에 담아주는 것 같음!
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # 평균 어텐션 맵 계산
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        # 상태 초기화
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        # 초기화 메소드
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
    # 이 함수까지 upper class. 
class AttentionControlEdit(AttentionStore, abc.ABC):
    # 어텐션 맵을 수정하는 클래스, AttentionStore를 상속
    def step_callback(self, x_t):
        # 각 스텝 후 호출되는 콜백 메소드, 로컬 블렌딩 적용
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        # 자기 자신에 대한 어텐션을 대체하는 메소드
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        # 다른 토큰에 대한 어텐션을 대체하는 메소드, 구현 필요
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 어텐션 수정 로직
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size) # h: head의 개수! 
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:] # 왜 첫번째 batch 사이즈는 분리하는 것이지? 
            '''
            -> 이해했다. 그 이유는 다름이 아닌 이 본 논문의 테스크가 두 개의 프롬프트를 사용해서 바뀐 단어만 바꾸는 editing task이기 때문에 \
            두 개의 prompt가 필요하고 한 prompt는 배경을 생성하는데 쓰이고 다른 하나의 prompt는 바뀐 단어를 명시하는데 쓰이기 때문에 \ 
            prompt도 두 개 따라서 batch 가 두개 하나는 base 하나는 replace가 되는 것이다. 
            '''

            # 아래 두 조건문은 cross attention을 replace할 것인지, self-attention을 replace할 것인지를 판단하는 것. 
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step] # replace할 것에 대한 mask! 
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                # attn_replace_new에는 replace_cross_attention함수가 적용된, 수정한 attention이 들어있을 것이다! 
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        # 결국 수정된 attention 값을 출력한다. 
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        # 초기화 메소드
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        # self.batch_size: 입력 prompt의 개수이다. (prompt_base, Prompt_replace.. )
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        # self.cross_replace_alpha: replace_token에 대한 alpha mask를 생성한다! 
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        # self.num_self_replace: self_replace할 step의 시작과 끝을 지정한다. 
        self.local_blend = local_blend
        # self.local_blend: local_bland라는 기능을 하용할지에 대한 bool값이다. (주로 사용하지 않는 것 같다.)

class AttentionReplace(AttentionControlEdit):
    # 어텐션 맵을 완전히 대체하는 클래스
    def replace_cross_attention(self, attn_base, att_replace):
        # 다른 토큰에 대한 어텐션 대체 로직
        print("type attn base",type(attn_base))
        print("shape(attn base)", attn_base.shape)
        print("shape(mapper)", mapper.shape)
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        # attention base를 수정해서 출력함! 
        # 생각해보니 attn_base도 중요한 이유가 결국 다시 생성할 때도 가장 많은 부분을 사용하는 것이 attn_base이다 attn_replace는 특정 부위에만 적용될 것이다. 
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        # 초기화 메소드
        # self.local_blend: local_bland라는 기능을 하용할지에 대한 bool값이다. (주로 사용하지 않는 것 같다.)
        # self.num_self_replace: self_replace할 step의 시작과 끝을 지정한다. 
        # self.cross_replace_alpha: replace_token에 대한 alpha mask를 생성한다! 
        # self.batch_size: 입력 prompt의 개수이다. (prompt_base, Prompt_replace.. )
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):
    # 어텐션 맵을 세밀하게 조정하는 클래스
    def replace_cross_attention(self, attn_base, att_replace):
        # 다른 토큰에 대한 어텐션 조정 로직
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        # 초기화 메소드
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    # 어텐션 가중치를 재조정하는 클래스
    def replace_cross_attention(self, attn_base, att_replace):
        # 다른 토큰에 대한 어텐션 가중치 재조정 로직
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        # 초기화 메소드
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    # 특정 단어의 어텐션 가중치를 조정하기 위한 "이퀄라이저" 생성 함수
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)  # 단어 인덱스 추출
        equalizer[:, inds] = values  # 이퀄라이저 업데이트
    return equalizer
