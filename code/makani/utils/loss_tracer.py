from makani.utils.losses import LossHandler
import torch

class TracerLossHandler(LossHandler):
    def __init__(self, params, tracer_channel_weights=None):
        super(TracerLossHandler, self).__init__(params)
        # Override the channel weights to match tracer output channels (87)
        if tracer_channel_weights is None:
            tracer_channel_weights = torch.ones(params.N_tracer_in_channels // (params.n_history + 1), dtype=torch.float32)
        else:
            tracer_channel_weights = torch.Tensor(tracer_channel_weights).float()

        # Normalize the weights
        tracer_channel_weights = tracer_channel_weights.reshape(1, -1, 1, 1)
        tracer_channel_weights = tracer_channel_weights / torch.sum(tracer_channel_weights)

        # Replace the parent's channel_weights buffer
        self.register_buffer("channel_weights", tracer_channel_weights)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, inp: torch.Tensor):
        # Same as before but now chw has 87 channels
        chw = self.channel_weights

        if self.training:
            chw = (chw * self.multistep_weight).reshape(1, -1)
        else:
            chw = chw.reshape(1, -1)
        return self.loss_obj(prd, tar, chw)
