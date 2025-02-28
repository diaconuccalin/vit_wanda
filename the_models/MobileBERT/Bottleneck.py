from torch import nn

from the_models.MobileBERT.BottleneckLayer import BottleneckLayer


class Bottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.input = BottleneckLayer(config)
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)

    def forward(self, hidden_states):
        layer_input = self.input(hidden_states)
        if self.use_bottleneck_attention:
            return [layer_input] * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (
                shared_attention_input,
                shared_attention_input,
                hidden_states,
                layer_input,
            )
        else:
            return (hidden_states, hidden_states, hidden_states, layer_input)
