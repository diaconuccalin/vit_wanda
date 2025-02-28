from torch import nn

from the_models.MobileBERT.MobileBertEmbeddings import NORM2FN


class FFNOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs
