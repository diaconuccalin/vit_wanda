import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class WrappedLayer:
    def __init__(self, layer, layer_id=0, layer_name="none", p_norm=2):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.n_samples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name
        self.p_norm = p_norm

    def add_batch(self, inp):
        assert inp.shape[-1] == self.columns

        inp = inp.reshape((-1, self.columns))
        tmp = inp.shape[0]
        inp = inp.t()

        if self.p_norm == 2:
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / tmp
        elif self.p_norm == 1:
            self.scaler_row += torch.norm(inp, p=1, dim=1) ** 1 / tmp

        if torch.isinf(self.scaler_row).sum() > 0:
            print("encountered torch.isinf error")
            raise ValueError
