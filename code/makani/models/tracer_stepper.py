import torch
import torch.nn as nn


class TracerMultiStepWrapper(nn.Module):
    """
    Multi-step autoregressive wrapper for the tracer head.

    Train:
        returns concatenated raw outputs over future steps
        if predict_delta=True, those raw outputs are deltas

    Eval:
        returns a single-step raw output
        if predict_delta=True, that raw output is a delta
    """
    def __init__(self, params, base_model):
        super().__init__()
        self.params = params
        self.base_model = base_model
        self.preprocessor = base_model.preprocessor
        self.n_future = params.n_future
        self.predict_delta = getattr(params, "predict_delta", False)
        self.c_out = params.N_tracer_out_channels

    def _current_state(self, tracer_state):
        """
        Extract the latest tracer field from history.
        tracer_state: [B, C*(n_hist+1), H, W] or [B, C, H, W]
        """
        if tracer_state.shape[1] > self.c_out:
            return tracer_state[:, -self.c_out:, ...]
        return tracer_state

    def _step(self, tracer_state, physical_features):
        """
        tracer_state: full flattened history [B, C*(n_hist+1), H, W]
        """
        current = self._current_state(tracer_state)   # only for delta/state update

        # encoder should see the full history
        tracer_features = self.base_model.tracer_features(tracer_state)

        fused = torch.cat([physical_features, tracer_features], dim=1)
        raw_pred = self.base_model.tracer_sfno(fused)

        return current, raw_pred

    def _forward_train(self, tracer_input, physical_features):
        tracer_state = tracer_input
        preds = []

        for step in range(self.n_future + 1):
            current, raw_pred = self._step(tracer_state, physical_features)

            # raw_pred is delta when predict_delta=True
            if self.predict_delta:
                next_state = current + raw_pred
                pred = raw_pred
            else:
                next_state = raw_pred
                pred = raw_pred

            preds.append(pred)

            if step < self.n_future:
                tracer_state = self.preprocessor.append_history(
                    tracer_state, next_state, step
                )

        return torch.cat(preds, dim=1)

    def _forward_eval(self, tracer_input, physical_features):
        _, raw_pred = self._step(tracer_input, physical_features)
        return raw_pred

    def forward(self, tracer_input, physical_features):
        if self.training:
            return self._forward_train(tracer_input, physical_features)
        return self._forward_eval(tracer_input, physical_features)