from torch import nn

from the_models.MobileBERT.FFNOutput import FFNOutput
from the_models.MobileBERT.MobileBertIntermediate import MobileBertIntermediate


class FFNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def forward(self, hidden_sites):
        intermediate_output = self.intermediate(hidden_sites)
        layer_outputs = self.output(intermediate_output, hidden_sites)
        return layer_outputs
