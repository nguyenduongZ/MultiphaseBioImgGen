import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing_extensions import Final
from collections.abc import Sequence
from monai.utils.misc import ensure_tuple_rep
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.module import optional_import, look_up_option
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer, PatchMergingV2, PatchMerging

rearrange, _ = optional_import("einops", name="rearrange")

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self, 
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        super().__init__()
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        
        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        
        self.normalize = normalize
        
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )
        
        self.encoder = nn.ModuleList(
            [
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=feature_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True
                ),
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True                    
                ),
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=2 * feature_size,
                    out_channels=2 * feature_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True                    
                ),
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=4 * feature_size,
                    out_channels=4 * feature_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True                    
                ),
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=16 * feature_size,
                    out_channels=16 * feature_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True                    
                )
            ]
        )
        
        self.decoder = nn.ModuleDict({
            "clf": nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(1),
                nn.Linear(16 * feature_size, 256),
                nn.GELU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 4)
            ),
            
            "seg": nn.ModuleList([
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=16 * feature_size,
                    out_channels=8 * feature_size,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True    
                ),
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=8 * feature_size,
                    out_channels=4 * feature_size,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True    
                ),
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=4 * feature_size,
                    out_channels=2 * feature_size,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True    
                ),
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=2 * feature_size,
                    out_channels=feature_size,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True    
                ),
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True    
                ),
            ])
        })

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)



    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        
        # Encoder
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder[0](x_in)
        enc1 = self.encoder[1](hidden_states_out[0])
        enc2 = self.encoder[2](hidden_states_out[1])
        enc3 = self.encoder[3](hidden_states_out[2])
        enc4 = self.encoder[4](hidden_states_out[4])
        
        # Decoder
        dec3 = self.decoder["seg"][0](enc4, hidden_states_out[3])
        dec2 = self.decoder["seg"][1](dec3, enc3)
        dec1 = self.decoder["seg"][2](dec2, enc2)
        dec0 = self.decoder["seg"][3](dec1, enc1)
        out = self.decoder["seg"][4](dec0, enc0)
        
        # z_feat
        pooled = self.decoder["clf"][0](enc4)
        pooled = self.decoder["clf"][1](pooled)
        
        logits = self.decoder["clf"][2:](pooled)
        masks = self.out(out)

        return {
            "category": logits,
            "z_feat": pooled,
            "semantic": masks
        }