from torch import nn

from the_models.MobileBERT.Bottleneck import Bottleneck
from the_models.MobileBERT.FFNLayer import FFNLayer
from the_models.MobileBERT.MobileBertAttention import MobileBertAttention
from the_models.MobileBERT.MobileBertIntermediate import MobileBertIntermediate
from the_models.MobileBERT.MobileBertOutput import MobileBertOutput


class MobileBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks

        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        if config.num_feedforward_networks != 1:
            self.ffn = nn.ModuleList(
                [FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)]
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(
                hidden_states
            )
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        outputs = (layer_output,) + outputs
        return outputs
