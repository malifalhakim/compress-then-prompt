import os
from pathlib import Path

from transformers import OPTForCausalLM, LlamaForCausalLM

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


class LLamaPromptTuningMixin:

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )

    # def forward_with_soft_prompt(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ):
    #     if input_ids is not None:
    #         inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
    #             self.device
    #         )

    #     if labels is not None:
    #         labels = self._extend_labels(labels).to(self.device)

    #     if attention_mask is not None:
    #         attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

    #     # Drop most of the args for now
    #     return super().forward(
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         inputs_embeds=inputs_embeds,
    #         labels=labels,
    #         use_cache=use_cache,
    #         return_dict=return_dict,
    #     )
    
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            print(model)
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )
        return model


    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path

        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")


    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.model.embed_tokens.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, self.config.hidden_size)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)


    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds


    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )


    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )


    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")


    def save_model(self, path: str):
        self.model.save_pretrained(path)

        # input_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,

# def shift_logits_and_labels(logits,
#                             loss_ids,
#                             reference_ids):

#     r"""
#     Left shift the label, and make label of the positions that are
#     not loss position to -100, which is the ignore index in pytorch's
#     loss function.
#     Args:
#         logits (:obj:`torch.Tensor`):
#         batch (:obj:`InputFeatures`): The input features of batchified data sequences.
#     Returns:
#         shift_logits (:obj:`torch.Tensor`):
#         shift_input_ids (:obj:`List[int]`):
#     """

#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_loss_ids = loss_ids[..., 1:].contiguous()
#     shift_input_ids = reference_ids[..., 1:].contiguous()
#     # shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100)
#     return shift_logits, shift_input_ids

class LLamaPromptTuningLM(LLamaPromptTuningMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)