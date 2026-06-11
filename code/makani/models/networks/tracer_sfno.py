import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp
from modulus.distributed.mappings import scatter_to_parallel_region, gather_from_parallel_region
from makani.utils import comm
from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet
from makani.models.common import EncoderDecoder
from makani.models.preprocessor import Preprocessor2D

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class CAMNetWithTracer(nn.Module):
    def __init__(self, inp_shape, out_shape, inp_chans, out_chans, **params):
        super().__init__()
        params = AttrDict(params)
        self.preprocessor = Preprocessor2D(params)
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp_chans = inp_chans               # physical input channels
        self.out_chans = out_chans               # output tracer channels
        self.embed_dim = params["embed_dim"]
        self.inp_chans_tracer = params["N_tracer_in_channels"]   # tracer input channels
        self.out_chans_tracer = params["N_tracer_out_channels"]    # output tracer channels
        # Backbone: physical SFNO
        self.physical_sfno = SphericalFourierNeuralOperatorNet(
            inp_chans=self.inp_chans,
            out_chans=self.out_chans,
            num_layers=8,
            scale_factor=params["scale_factor"],
            embed_dim=self.embed_dim,
            spectral_transform=params.get("spectral_transform", "sht"),
            model_grid_type=params.get("model_grid_type", "equiangular"),
            filter_type=params.get("filter_type", "linear"),
            operator_type=params.get("operator_type", "dhconv"),
            **{k: v for k, v in params.items() if k in [
                'big_skip', 'factorization', 'rank', 'separable',
                'normalization_layer', 'hard_thresholding_fraction']}
        )

        # Tracer encoder
        self.tracer_encoder = EncoderDecoder(
            num_layers=1,
            input_dim=self.inp_chans_tracer,
            output_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            act_layer=nn.GELU,
            input_format="nchw"
        )

        # Tracer SFNO head
        self.tracer_sfno = SphericalFourierNeuralOperatorNet(
            inp_chans=2 * self.embed_dim,
            out_chans=self.out_chans_tracer,
            num_layers=4,
            embed_dim=self.embed_dim,
            spectral_transform=params.get("spectral_transform", "sht"),
            model_grid_type=params.get("model_grid_type", "equiangular"),
            filter_type=params.get("filter_type", "linear"),
            operator_type=params.get("operator_type", "dhconv"),
            pos_embed=params.get("pos_embed", "none"),  # ✅ add this line
            **{k: v for k, v in params.items() if k in [
                'big_skip', 'factorization', 'rank', 'separable']}
        )
        
        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.tracer_encoder.apply(_init)
        self.tracer_sfno.apply(_init)

    def forward(self, physical_input, tracer_input):
        inpa = self.preprocessor.append_unpredicted_features(physical_input)
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)
        inpans = self.preprocessor.add_static_features(inpan)
        with torch.set_grad_enabled(not self.training):
            physical_features = self.physical_features(inpans)
        tracer_features = self.tracer_encoder(tracer_input)
        fused = torch.cat([physical_features, tracer_features], dim=1) #sum the features
        # Predict tracer outputs
        return self.tracer_sfno(fused)

    def predict_tracer(self, fused_tensor):
        torch.cuda.empty_cache()        # frees unused cached blocks
        torch.cuda.ipc_collect()        # releases inter-process handles (useful under DDP)
        torch.cuda.synchronize()        # ensure all ops complete
        return self.tracer_sfno(fused_tensor)
    
    def physical_features(self, inpans):
        with torch.set_grad_enabled(not self.training):
            if comm.get_size("fin") > 1:
                inpans = scatter_to_parallel_region(inpans, 1, "fin")
            if self.physical_sfno.checkpointing >= 1:
                encoded = checkpoint(self.physical_sfno.encoder, inpans, use_reentrant=False)
            else:
                encoded = self.physical_sfno.encoder(inpans)
            if hasattr(self.physical_sfno, "pos_embed"):
                pos_embed = self.physical_sfno.pos_embed
                if pos_embed.type == "frequency":
                    pos_embed = torch.stack([
                        pos_embed[0],
                        nn.functional.pad(pos_embed[1], (1, 0), "constant", 0)
                    ], dim=-1)
                    with amp.autocast(enabled=False):
                        pos_embed = self.physical_sfno.itrans_up(torch.view_as_complex(pos_embed))
                encoded = encoded + pos_embed
            encoded = self.physical_sfno.pos_drop(encoded)
            features = self.physical_sfno._forward_features(encoded)
            return features

    def predict_physical(self,inpans):
        with torch.set_grad_enabled(not self.training):
            physical_output = self.physical_sfno(inpans)
        return physical_output
    
    def tracer_features(self, tracer_input):
        return self.tracer_encoder(tracer_input)


        
