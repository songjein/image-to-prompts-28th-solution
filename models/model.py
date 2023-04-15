from typing import Dict

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel, GPT2LMHeadModel, SwinModel


class VisionModel(nn.Module):
    def __init__(
        self, model_name: str, out_features: int, frozen_backbone: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.model = self._create_model()

        self.output_dimension = self._get_output_dimension()
        self.projection_features = nn.Linear(
            in_features=self.output_dimension,
            out_features=out_features,
        )

        #self.clip_projection = nn.Linear(
        #    in_features=self.output_dimension,
        #    out_features=384,
        #)

        if frozen_backbone:
            for p in self.model.parameters():
                p.required_grad = False

    def _create_model(self):
        #config = CLIPVisionConfig.from_pretrained(self.model_name)
        model = SwinModel.from_pretrained(self.model_name)
        return model

    def _get_output_dimension(self):
        return self.model.config.hidden_size

    '''
    def get_clip_embedding(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.model(pixel_values)

        if not isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.pooler_output
        embeddings = embeddings.reshape(batch_size, self.output_dimension)
        embeddings = self.clip_projection(embeddings)

        return embeddings
    '''

    def forward(self, pixel_values: torch.tensor):
        batch_size = pixel_values.shape[0]
        embeddings = self.model(pixel_values)

        if not isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.pooler_output

        embeddings = embeddings.reshape(batch_size, self.output_dimension)
        embeddings = self.projection_features(embeddings)

        return embeddings


class LanguageModel(GPT2LMHeadModel):
    def __init__(self, config):
        super(LanguageModel, self).__init__(config)

        self.n_embd = self.config.n_embd

    def forward(
        self,
        input_ids=None,
        image_token_embeddings=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_token_mask=None,
    ):
        # Project image embeddings to token embeddings
        if image_token_embeddings is not None and image_token_mask is not None:
            inputs_embeds = self.transformer.wte(input_ids)
            ind = image_token_mask.nonzero(as_tuple=True)
            # token 개수 만큼으로 reshape
            image_token_embeddings = image_token_embeddings.reshape(-1, self.n_embd)
            inputs_embeds[ind] = image_token_embeddings.type(inputs_embeds.dtype)
            input_ids = None

        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def generate(
        self,
        input_ids=None,
        image_token_embeddings=None,
        max_length=None,
        **generate_kwargs
    ):
        # Project image embeddings to token embeddings
        if image_token_embeddings is not None:
            batch_size = image_token_embeddings.shape[0]
            if input_ids is not None:
                inputs_embeds = self.transformer.wte(input_ids)
                image_token_embeddings = image_token_embeddings.reshape(batch_size, -1, self.n_embd)
                #inputs_embeds[:, 0] = image_token_embeddings.type(inputs_embeds.dtype)
                inputs_embeds = image_token_embeddings
                input_ids = None
            else:
                image_token_embeddings = image_token_embeddings.reshape(batch_size, -1, self.n_embd)
                inputs_embeds = image_token_embeddings

            input_ids = None
            return super().generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                max_length=max_length,
                **generate_kwargs
            )
        else:
            return super().generate(
                input_ids=input_ids, max_length=max_length, **generate_kwargs
            )


class EncoderDecoder(nn.Module):
    def __init__(
        self, encoder: VisionModel, decoder: LanguageModel, config: Dict[str, bool]
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        if config["encoder_frozen"]:
            for p in self.encoder.model.parameters():
                p.requires_grad = False

        if config["decoder_frozen"]:
            for p in self.decoder.parameters():
                p.requires_grad = False

    def forward(self, *args, **kwargs):
        if "pixel_values" in kwargs:
            pixel_values = kwargs.pop("pixel_values")
            kwargs["image_token_embeddings"] = self.encoder(pixel_values)
        return self.decoder.forward(*args, **kwargs, return_dict=True)

    def generate(self, *args, **kwargs):
        if "pixel_values" in kwargs:
            pixel_values = kwargs.pop("pixel_values")
            kwargs["image_token_embeddings"] = self.encoder(pixel_values)
        return self.decoder.generate(*args, **kwargs, return_dict_in_generate=True)
