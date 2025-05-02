import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import torch.nn as nn

from .block import PatchEmbed, PatchMerging, trunc_normal_, BasicLayer

class SwinUnetEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 1],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False        
    ):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.pathes_resolution
        
        # Absolute Position Embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build Encoder Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers -1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            
    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)
        
        # Absolute Position Embedding
        if self.ape:
            x = x + self.absolute_pos_embed

        # Position Dropout
        x = self.pos_drop(x)
        
        # Forward through encoder layers
        skip_connections = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Store output of each layer before downsampling for skip connections
            if i < self.num_layers - 1:  # Skip the bottleneck
                H, W = self.patches_resolution[0] // (2 ** i), self.patches_resolution[1] // (2 ** i)
                skip = x.view(-1, H, W, int(self.embed_dim * 2 ** i))
                skip_connections.append(skip)

        # Reshape the final output (bottleneck)
        H, W = self.patches_resolution[0] // (2 ** (self.num_layers - 1)), self.patches_resolution[1] // (2 ** (self.num_layers - 1))
        x = x.view(-1, H, W, int(self.embed_dim * 2 ** (self.num_layers - 1)))

        return x, skip_connections

    def get_output_resolutions(self):
        resolutions = []
        for i in range(self.num_layers):
            H = self.patches_resolution[0] // (2 ** i)
            W = self.patches_resolution[1] // (2 ** i)
            resolutions.append((H, W))
        return resolutions