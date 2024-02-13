# 필요한 라이브러리들을 가져옵니다.
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm

# 이미지 아래에 텍스트를 추가하는 함수
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    # 이미지의 크기를 가져옵니다.
    h, w, c = image.shape
    # 텍스트를 위한 추가적인 공간을 계산합니다.
    offset = int(h * .2)
    # 새로운 이미지를 생성합니다.
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    # OpenCV를 사용하여 텍스트를 이미지에 추가합니다.
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

# 여러 이미지를 시각화하는 함수
def view_images(images, num_rows=1, offset_ratio=0.02):
    # 이미지가 리스트인지 확인하고 필요한 경우 형태를 조정합니다.
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    # 비어있는 이미지를 생성하여 그리드를 채웁니다.
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    # 이미지 그리드를 생성합니다.
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    # PIL 라이브러리를 사용하여 이미지를 표시합니다.
    pil_img = Image.fromarray(image_)
    display(pil_img)

# 디퓨전 과정에서 한 스텝을 진행하는 함수
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    # low_resource 모드에 따라 노이즈 예측을 조정합니다.
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents

# 잠재 벡터(latent)를 이미지로 변환하는 함수
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

# 잠재 벡터를 초기화하는 함수
def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

# 텍스트를 이미지로 변환하는 함수 (Latent Diffusion Model 기반)
@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    # 모델에 어텐션 컨트롤러를 등록합니다.
    register_attention_control(model, controller)
    # 이미지의 높이와 너비를 설정합니다.
    height = width = 256
    # 배치 사이즈를 설정합니다.
    batch_size = len(prompt)
    
    # 조건 없는(빈 문자열) 입력과 조건 있는(프롬프트) 입력을 토크나이저를 통해 처리합니다.
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    # 잠재 벡터를 초기화합니다.
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    # 조건 없는 입력과 조건 있는 입력을 결합합니다.
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    # 디퓨전 스케줄러를 설정합니다.
    model.scheduler.set_timesteps(num_inference_steps)
    # 디퓨전 과정을 진행합니다.
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    # 생성된 이미지를 얻습니다.
    image = latent2image(model.vqvae, latents)
   
    return image, latent

# Stable 버전의 Latent Diffusion Model을 사용하여 텍스트를 이미지로 변환하는 함수
@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    # 모델에 어텐션 컨트롤러를 등록합니다.
    register_attention_control(model, controller)
    # 이미지의 높이와 너비를 설정합니다.
    height = width = 512
    # 배치 사이즈를 설정합니다.
    batch_size = len(prompt)

    # 프롬프트를 토크나이저를 통해 처리합니다.
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # 텍스트 임베딩을 생성합니다.
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    # 조건 없는 입력과 조건 있는 입력을 결합합니다.
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    # 잠재 벡터를 초기화합니다.
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # 디퓨전 스케줄러를 설정합니다.
    model.scheduler.set_timesteps(num_inference_steps)
    # 디퓨전 과정을 진행합니다.
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    # 생성된 이미지를 얻습니다.
    image = latent2image(model.vae, latents)
  
    return image, latent

# 모델에 어텐션 컨트롤을 등록하는 함수입니다.
def register_attention_control(model, controller):
    # 주어진 모델의 어텐션 레이어에 대한 커스텀 forward 함수를 정의합니다.
    def ca_forward(self, place_in_unet):
        # self.to_out이 ModuleList인 경우 첫 번째 요소를 선택합니다.
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        # 커스텀 forward 함수 정의
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,):
            # 교차 어텐션 여부를 확인합니다.
            is_cross = encoder_hidden_states is not None
            
            # 원본 히든 상태를 저장합니다.
            residual = hidden_states

            # spatial_norm과 group_norm을 적용합니다.
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            # 쿼리, 키, 값 텐서를 계산합니다.
            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            # 어텐션 스코어를 계산하고, 사용자 정의 컨트롤러를 적용합니다.
            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            # 새로운 히든 상태를 계산합니다.
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)
            hidden_states = to_out(hidden_states)

            # 입력 차원을 다시 조정합니다.
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            # 잔차 연결을 추가합니다.
            if self.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    # 더미 컨트롤러를 생성하는 클래스입니다. 
    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0

    # 컨트롤러가 None인 경우 더미 컨트롤러를 사용합니다.
    if controller is None:
        controller = DummyController()

    # 모델의 어텐션 레이어에 커스텀 forward 함수를 적용합니다.
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    # 어텐션 레이어의 개수를 세는 변수를 초기화합니다.
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    # 모델의 각 하위 네트워크에 대해 register_recr 함수를 호출합니다.
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    # 컨트롤러의 어텐션 레이어 개수를 업데이트합니다.
    controller.num_att_layers = cross_att_count


def get_word_inds(text: str, word_place: int, tokenizer):
    # 텍스트를 단어별로 분할합니다.
    split_text = text.split(" ")
    # word_place가 문자열인 경우, 해당 단어가 텍스트 내에서 등장하는 모든 위치를 찾습니다.
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    # word_place가 정수인 경우, 리스트로 변환합니다.
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    # word_place에 해당하는 단어의 위치를 찾습니다.
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

# 특정 시간에 대한 어텐션 가중치를 업데이트하는 함수
def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    # bounds가 단일 숫자인 경우, 범위를 설정합니다.
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    # word_inds가 None인 경우, 모든 단어를 대상으로 합니다.
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    # 지정된 시간 범위 내에서 word_inds에 해당하는 단어의 가중치를 조절합니다.
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

# 어텐션 가중치를 얻는 함수
def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    # cross_replace_steps가 단일 숫자인 경우, 기본값 설정
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    # 어텐션 가중치를 초기화합니다.
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    # 각 프롬프트에 대해 어텐션 가중치를 설정합니다.
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    # cross_replace_steps에 정의된 대로 특정 단어에 대한 가중치를 조정합니다.
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words