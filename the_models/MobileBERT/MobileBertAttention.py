from torch import nn

from the_models.MobileBERT.MobileBertSelfAttention import MobileBertSelfAttention
from the_models.MobileBERT.MobileBertSelfOutput import MobileBertSelfOutput


class MobileBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MobileBertSelfAttention(config)
        self.output = MobileBertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        layer_input,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            query_tensor,
            key_tensor,
            value_tensor,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        attention_output = self.output(self_outputs[0], layer_input)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs
