from torch import nn

from the_models.MobileBERT.activations import ACT2FN


class MobileBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.intermediate_act_fn(layer_outputs)
        return layer_outputs
