from typing import Callable, Optional
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from accelerate.logging import get_logger

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.models.attention import Attention
from diffusers.utils.import_utils import is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

logger = get_logger(__name__)
import sys

class CustomDiffusionAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        """
        Process attention mechanism for diffusion models without xformers.

        Args:
            attn (Attention): Attention module.
            hidden_states: Hidden states.
            encoder_hidden_states: Encoder hidden states. Defaults to None.
            attention_mask: Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Processed hidden states.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :]*0.
            key = detach*key + (1-detach)*key.detach()
            value = detach*value + (1-detach)*value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class CustomDiffusionPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for custom diffusion model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        modifier_token: list of new modifier tokens added or to be added to text_encoder
        modifier_token_id: list of id of new modifier tokens added or to be added to text_encoder
    """
    # _optional_components = ["safety_checker", "feature_extractor", "modifier_token"]
    _optional_components = ["feature_extractor", "modifier_token"]

    # thum_code: safety_checker : false  / original : True
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
        modifier_token: list = [],
        modifier_token_id: list = [],
    ):
        super().__init__(vae,
                         text_encoder,
                         tokenizer,
                         unet,
                         scheduler,
                         safety_checker,
                         feature_extractor,
                         requires_safety_checker=requires_safety_checker)

        # change attn class
        self.modifier_token = modifier_token
        self.modifier_token_id = modifier_token_id

    def add_token(self, initializer_token):
        """
        Add new modifier tokens to the text_encoder, along with their corresponding initializer tokens.

        Args:
            initializer_token (list): List of initializer tokens to be paired with the new modifier tokens.
        """
        initializer_token_id = []
        for modifier_token_, initializer_token_ in zip(self.modifier_token, initializer_token):
            # Add the placeholder token in tokenizer
            num_added_tokens = self.tokenizer.add_tokens(modifier_token_)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token_}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = self.tokenizer.encode([initializer_token_], add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            self.modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token_))
            initializer_token_id.append(token_ids[0])
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        for (x, y) in zip(self.modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

    def save_pretrained(self, save_path, freeze_model="crossattn_kv", save_text_encoder=False, all=False):
        """
        Save the model's state and optionally freeze specific components.

        Args:
            save_path (str): Path to save the model.
            freeze_model (str): Freezing option for the U-Net model.
                "crossattn_kv" freezes the cross-attention key and value projections in the U-Net model.
                "crossattn" freezes the entire cross-attention mechanism in the U-Net model.
            save_text_encoder (bool): Whether to save the text_encoder.
            all (bool): Whether to save all components.
        """
        if all:
            super().save_pretrained(save_path)
        else:
            delta_dict = {'unet': {}, 'modifier_token': {}}
            if self.modifier_token is not None:
                for i in range(len(self.modifier_token_id)):
                    learned_embeds = self.text_encoder.get_input_embeddings().weight[self.modifier_token_id[i]]
                    delta_dict['modifier_token'][self.modifier_token[i]] = learned_embeds.detach().cpu()
            if save_text_encoder:
                delta_dict['text_encoder'] = self.text_encoder.state_dict()
            for name, params in self.unet.named_parameters():
                if freeze_model == "crossattn":
                    if 'attn2' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                elif freeze_model == "crossattn_kv":
                    if 'attn2.to_k' in name or 'attn2.to_v' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                else:
                    raise ValueError(
                            "freeze_model argument only supports crossattn_kv or crossattn"
                        )
            torch.save(delta_dict, save_path)

    def load_model(self, save_path, compress=False):
        """
        Load a pre-trained model from a specified path.

        Args:
            save_path (str): Path to the saved model.
            compress (bool): Whether to apply compression during loading.
        """
        st = torch.load(save_path)
        if 'text_encoder' in st:
            self.text_encoder.load_state_dict(st['text_encoder'])
        if 'modifier_token' in st:
            modifier_tokens = list(st['modifier_token'].keys())
            modifier_token_id = []
            for modifier_token in modifier_tokens:
                num_added_tokens = self.tokenizer.add_tokens(modifier_token)
                if num_added_tokens == 0:
                    raise ValueError(
                        f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )
                modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token))
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            for i, id_ in enumerate(modifier_token_id):
                token_embeds[id_] = st['modifier_token'][modifier_tokens[i]]

        for name, params in self.unet.named_parameters():
            if 'attn2' in name:
                if compress and ('to_k' in name or 'to_v' in name):
                    params.data += st['unet'][name]['u']@st['unet'][name]['v']
                elif name in st['unet']:
                    params.data.copy_(st['unet'][f'{name}'])

# The provided code seems to be incomplete, and the last part of the code is missing.
# The last part of the code may include additional methods, usage examples, or other functionality.
