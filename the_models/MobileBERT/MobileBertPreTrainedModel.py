from torch import nn

from the_models.MobileBERT.ManualLayerNorm import ManualLayerNorm
from the_models.MobileBERT.MobileBertConfig import MobileBertConfig
from the_models.MobileBERT.NoNorm import NoNorm
from the_models.MobileBERT.PreTrainedModel import PreTrainedModel
from the_models.MobileBERT.utils import load_tf_weights_in_mobilebert

MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {}


class MobileBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = MobileBertConfig
    pretrained_model_archive_map = MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_mobilebert
    base_model_prefix = "Mobilebert"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, NoNorm, ManualLayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
