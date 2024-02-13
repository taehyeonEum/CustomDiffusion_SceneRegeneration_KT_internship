    # Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np

# 점수 매개변수를 설정하는 클래스입니다.
class ScoreParams:

    def __init__(self, gap, match, mismatch):
        # 갭, 매치, 미스매치에 대한 점수를 초기화합니다.
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        # 두 문자가 일치하지 않을 때 점수를 반환합니다.
        if x != y:
            return self.mismatch
        else: 
            return self.match

# 배열(매트릭스)을 생성하는 함수입니다.   
def get_matrix(size_x, size_y, gap):
    # NumPy를 사용하여 2차원 0으로 채워진 배열을 만듭니다.
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    # 첫 번째 행과 열을 갭 점수로 초기화합니다.
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix

# 역추적 매트릭스를 생성하는 함수입니다.
def get_traceback_matrix(size_x, size_y):
    # 2차원 0으로 채워진 배열을 만듭니다.
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    # 첫 번째 행과 열을 초기화합니다.
    matrix[0, 1:] = 1  # 왼쪽 이동
    matrix[1:, 0] = 2  # 위쪽 이동
    matrix[0, 0] = 4  # 시작 지점
    return matrix

# 글로벌 정렬을 수행하는 함수입니다.
def global_align(x, y, score):
    # 점수 매트릭스와 역추적 매트릭스를 생성합니다.
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    # 각 셀에 대한 점수를 계산합니다.
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            # 역추적 매트릭스를 업데이트합니다.
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back

# 정렬된 시퀀스를 얻는 함수입니다.
def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    # 역추적 매트릭스를 따라가며 정렬된 시퀀스를 생성합니다.
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append('-')
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i-1])
            y_seq.append('-')
            i = i-1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)

# 매퍼를 생성하는 함수입니다.
def get_mapper(x: str, y: str, tokenizer, max_len=77):
    # 두 시퀀스를 토큰화합니다.
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    # 글로벌 정렬을 수행합니다.
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    # 정렬된 시퀀스와 매퍼를 얻습니다.
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    # 알파 값(토큰의 존재 여부)을 설정합니다.
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    # 매퍼를 설정합니다.
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas

# 정제 매퍼를 생성하는 함수입니다.
def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers, alphas = [], []
    # 각 프롬프트에 대해 매퍼를 생성합니다.
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)

# 특정 단어의 인덱스를 얻는 함수입니다.
def get_word_inds(text: str, word_place: int, tokenizer):
    # text는 prompt를, word_place는 인덱스, tokenizer는 말 그대로 tokenizer이다. 
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        # 입력으로 받은 단어의 tokenizer의 encoding 값이 들어있는 것 같다. encodeing과 decodeing을 동시에 진행하는 이유는 원래 그런 것 같다..
        # ['a', 'painting', 'of', 'a', 'squirrel', 'eating', 'a', 'burger']
        # ['a', 'painting', 'of', 'a', 'lion', 'eating', 'a', 'burger']
        cur_len, ptr = 0, 0
        # 각 단어의 인덱스를 찾습니다.
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

# 교체 매퍼를 생성하는 함수입니다.
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    # x, y는 한 문장의 prompt가 string 형식으로 저장되어있다. 
    words_x = x.split(' ')
    words_y = y.split(' ')
    # words_x, words_y에는 prompt가 띄어쓰기로 split 되어서 list 형태로 저장되어있다. 
    if len(words_x) != len(words_y):
        # 나 같은 경우에는 prompt도 같고 단어도 같지만 모델이 다르기 때문에 이 줄은 반드시 수정이 필요하다. 
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    # inds_replace에는 두 prompt에서 달랐던 단어의 index가 저장되어 있다. 
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]

    # sourece prompt의 단어 list와 index 그리고 tokenizer가 get_word_inds에 입력된다. 
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    # replace prompt의 단어 list와 index 그리고 tokenizer가 get_word_inds에 입력된다. 
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    # 매퍼를 생성합니다.
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1
    return torch.from_numpy(mapper).float()

# 교체 매퍼를 생성하는 함수입니다.
def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)
