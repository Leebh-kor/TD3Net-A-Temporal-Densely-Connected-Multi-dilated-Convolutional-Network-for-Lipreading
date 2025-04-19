import math
import warnings

import numpy as np
import timm
import torch
import torch.nn as nn
from einops import rearrange

from lipreading.modules import (
    BasicBlock,
    ResNet,
    TD3Net,
)


class FrameEmbedding(nn.Module):
    """ 
    input  : B, 1, T, H, W
    output : B, T, C 
    """
    def __init__(
        self, 
        in_channels = 1, 
        emb_size = 512, 
        backbone_type = 'efficientnetv2_s', 
        relu_type = 'prelu',
        use_pretrained = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        
        # Create backbone first to determine frontend output channels
        if 'resnet' in backbone_type:
            if use_pretrained:
                warnings.warn("Pretrained weights are only supported for EfficientNet models. Ignoring use_pretrained=True for ResNet.")
            self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            self.emb_dim = emb_size
            frontend_nout = 64  # ResNet expects 64 channels
            
        elif 'efficientnet' in backbone_type:
            # Create EfficientNet backbone
            backbone = timm.create_model(backbone_type, pretrained=use_pretrained)
            if hasattr(backbone, 'conv_stem'):
                frontend_nout = backbone.conv_stem.out_channels  # Get first layer's output channels
                backbone.conv_stem = nn.Identity()
                backbone.bn1 = nn.Identity()
            elif hasattr(backbone, 'conv1'):
                frontend_nout = backbone.conv1.out_channels  # Get first layer's output channels
                backbone.conv1 = nn.Identity()
                backbone.bn1 = nn.Identity()
            
            # Remove classifier head
            backbone.classifier = nn.Identity()
            self.trunk = backbone
            self.emb_dim = backbone.num_features
        
        # Common frontend with matched output channels
        frontend_relu = nn.PReLU(num_parameters=frontend_nout) if relu_type == 'prelu' else nn.SiLU()
        self.frontend = nn.Sequential(
            nn.Conv3d(in_channels, frontend_nout, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        pretrained_status = "with pretrained weights" if use_pretrained and 'efficientnet' in backbone_type else "without pretrained weights"
        print(f"Using backbone: {backbone_type} ({pretrained_status}), Frontend channels: {frontend_nout}, Output dimension: {self.emb_dim}")

    def forward(self, x):
        B, _, T, H, W = x.size()
        
        # Frontend 3D convolution
        x = self.frontend(x)
        
        # Reshape to 2D for backbone
        x = rearrange(x, "b c t h w -> (b t) c h w")
        
        # Backbone
        x = self.trunk(x)
        
        # Reshape back to temporal dimension
        x = rearrange(x, "(b t) c -> b c t", b=B)
        
        return x
    
class TD3Net_Lipreading(nn.Module):
    def __init__(
        self,
        emb_size,
        backbone_type,
        relu_type,
        growth_rate,
        num_layers,
        num_td2_blocks,
        out_block,
        block_comp,
        trans_comp_factor,
        kernel_size,
        is_cbr,
        dropout_p,
        use_td3_block=True,
        use_multi_dilation=True,
        use_bottle_layer=True,
        use_pretrained=False,
    ):
        super().__init__()
        self.emd_network = FrameEmbedding(
                                emb_size=emb_size,
                                backbone_type=backbone_type,
                                relu_type=relu_type,
                                use_pretrained=use_pretrained,
                                )
        emb_dim = self.emd_network.emb_dim
        
        # TD3Net configuration
        self.backend_network = TD3Net(
                    in_ch=emb_dim,
                    growth_rate=growth_rate,
                    num_layers=num_layers,
                    num_td2_blocks=num_td2_blocks,
                    out_block=out_block,
                    block_comp=block_comp,
                    trans_comp_factor=trans_comp_factor,
                    kernel_size=kernel_size,
                    is_cbr=is_cbr,
                    dropout_p=dropout_p,
                    relu_type=relu_type,
                    use_td3_block=use_td3_block,
                    use_multi_dilation=use_multi_dilation,
                    use_bottle_layer=use_bottle_layer,
                )
        self.cls_layer = nn.Linear(self.backend_network.out_ch, 500)
        
        # -- initialize
        self.initialize_weights_randomly()
        
    def forward(self, x, lengths=None):
        x = self.emd_network(x) # [B, C, T]
        out = self.backend_network(x)
        out = self.cls_layer(out)
            
        return out

    def initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)
        for name, m in self.named_modules():
            if 'trunk' in name:
                continue
            
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))


if __name__ == '__main__':
    # Model variants to test
    model_configs = [
        # TD3Net variants
        {'name': 'TD3Net (default)', 'use_td3_block': True, 'use_multi_dilation': True, 'use_bottle_layer': False},
        # {'name': 'TD3Net (no bottleneck)', 'use_td3_block': True, 'use_multi_dilation': True, 'use_bottle_layer': False},
        # TD2Net variants
        # {'name': 'TD2Net', 'use_td3_block': False, 'use_multi_dilation': True, 'use_bottle_layer': True},
        # {'name': 'TD2Net (no bottleneck)', 'use_td3_block': False, 'use_multi_dilation': True, 'use_bottle_layer': False},
        # Dense-TCN variants
        # {'name': 'Dense-TCN', 'use_td3_block': False, 'use_multi_dilation': False, 'use_bottle_layer': True},
        # {'name': 'Dense-TCN (no bottleneck)', 'use_td3_block': False, 'use_multi_dilation': False, 'use_bottle_layer': False},
    ]
    
    # Default input size
    input_size = (2, 1, 29, 88, 88)  # batch_size, channels, frames, height, width
    
    # Backbone types - use tf_efficientnetv2_s instead of efficientnetv2_s
    backbone_types = ['resnet', 'tf_efficientnetv2_s', 'tf_efficientnetv2_m', 'tf_efficientnetv2_l']  # This is the correct model name in timm
    # backbone_types = ['tf_efficientnetv2_s']
    # Configuration from td3net_config_base.yaml
    config = {
        'growth_rate': [52, 52, 52, 52],
        'num_layers': [16, 20, 24, 18],
        'num_td2_blocks': [1, 1, 1, 1],
        'out_block': [16, 20, 24, 18],
        'block_comp': [0.5, 0.5, 0.5, 1],
        'trans_comp_factor': [2, 2, 2, 1],
        'kernel_size': 3,
        'is_cbr': True,
        'dropout_p': 0.2,
        'use_pretrained': False  # Default value from config.py
    }
    
    print("\n=== Model Testing Started ===\n")
    
    for model_config in model_configs:
        print(f"\n[Model Variant: {model_config['name']}]")
        print(f"Configuration: use_td3_block={model_config['use_td3_block']}, "
              f"use_multi_dilation={model_config['use_multi_dilation']}, "
              f"use_bottle_layer={model_config['use_bottle_layer']}")
        
        for backbone in backbone_types:
            print(f"\n  Backbone: {backbone}")
            
            # Model initialization with config from yaml
            model = TD3Net_Lipreading(
                emb_size=512,
                backbone_type=backbone,
                relu_type='prelu',
                growth_rate=config['growth_rate'],
                num_layers=config['num_layers'],
                num_td2_blocks=config['num_td2_blocks'],
                out_block=config['out_block'],
                block_comp=config['block_comp'],
                trans_comp_factor=config['trans_comp_factor'],
                kernel_size=config['kernel_size'],
                is_cbr=config['is_cbr'],
                dropout_p=config['dropout_p'],
                use_td3_block=model_config['use_td3_block'],
                use_multi_dilation=model_config['use_multi_dilation'],
                use_bottle_layer=model_config['use_bottle_layer'],
                use_pretrained=True,
            )
            
            try:
                # Print results
                print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
                
                # Create dummy input
                dummy_input = torch.randn(input_size)
                
                # Forward pass test
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    try:
                        output = model(dummy_input)
                        print(f"Forward pass successful. Output shape: {output.shape}")
                    except Exception as e:
                        print(f"Forward pass failed: {str(e)}")
                
                # Print model summary using torchinfo with error handling
                try:
                    from torchinfo import summary
                    print("\nModel Summary:")
                    model_summary = summary(model, input_size=input_size, device='cpu', verbose=0)
                    print(model_summary)
                except Exception as e:
                    print(f"Failed to generate model summary: {str(e)}")
                
            except Exception as e:
                print(f"Error occurred: {str(e)}")
            
            print("\n" + "="*50)
    
    print("\n=== Model Testing Completed ===")
