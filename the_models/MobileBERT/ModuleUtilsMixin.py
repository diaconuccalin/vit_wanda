import os

import torch
from torch import Tensor, device, dtype


class ModuleUtilsMixin:
    """
    A few utilities for torch.nn.Modules, to be used as a mixin.
    """

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get number of (optionally, trainable) parameters in the module.
        """
        params = (
            filter(lambda x: x.requires_grad, self.parameters())
            if only_trainable
            else self.parameters()
        )
        return sum(p.numel() for p in params)

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError(
                "You need to install psutil (pip install psutil) to use memory tracing."
            )

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError(
                "You need to install psutil (pip install psutil) to use memory tracing."
            )

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (
            module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0
        )
        return None

    def add_memory_hooks(self):
        """Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.
        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero with `model.reset_memory_hooks_state()`
        """
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> dtype:
        return next(self.parameters()).dtype

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """type: torch.Tensor -> torch.Tensor"""
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        return encoder_extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: tuple, device: device
    ):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers):
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = (
                head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            )  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(
            dtype=self.dtype
        )  # switch to fload if need + fp16 compatibility
        return head_mask
