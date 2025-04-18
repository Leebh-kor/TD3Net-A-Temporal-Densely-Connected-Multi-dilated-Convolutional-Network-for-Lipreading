import torch
import torch.nn as nn
from einops import reduce

from .conv_modules import BasicConv, DBasicBlock, DenseBlock


class TD3Net(nn.Module):
    '''
    D3Net backbone implementation based on the paper:
    "Densely connected multidilated convolutional networks for dense prediction tasks", CVPR2021
    arXiv: https://arxiv.org/abs/2011.11844

    This network architecture consists of multiple TD3 blocks, each containing several TD2 blocks.
    The network progressively processes input features through dense connections and multi-dilated convolutions.

    Parameters:
    -----------
    in_ch : int
        Number of input channels (default: 512, typical for ResNet-18 output)
    growth_rate : List[int]
        Number of channels extracted from each layer (default: [36,36,36,36])
    num_layers : List[int]
        Number of layers in each TD2 Block (default: [5,5,5,5])
    num_td2_blocks : List[int]
        Number of TD2 blocks in each TD3 Block (default: [10,10,10,10])
    out_block : List[int]
        Controls channel reduction of TD2 block outputs (default: [5,5,5,5])
    block_comp : List[float]
        Compression factor for TD2 block outputs (default: [0.5,0.5,0.5,0.5])
    trans_comp_factor : List[int]
        Compression factor for TD3 block outputs (default: [2,2,2,1])
    kernel_size : int
        Size of convolutional kernels (default: 3)
    is_cbr : bool
        Whether to use post-activation function (default: True)
    dropout_p : float
        Dropout probability (default: 0.2)
    relu_type : str
        Type of ReLU activation (default: 'prelu')
    use_td3_block : bool
        Whether to use TD3 blocks (True) or only TD2 blocks (False) (default: True)
    use_multi_dilation : bool
        Whether to use multi-dilated convolutions (default: True)
    use_bottle_layer : bool
        Whether to use bottleneck layer in TD2 blocks (default: True)

    Architecture:
    ------------
    The network consists of multiple TD3 blocks, where each TD3 block contains:
    1. Multiple TD2 blocks with dense connections
    2. A transition layer for channel reduction
    Each TD2 block contains:
    1. Optional bottleneck layer for channel reduction
    2. Initial convolution layer
    3. Multi-dilated dense block or regular dense block
    4. Block reduction layer
    '''
    def __init__(
        self,
        in_ch=512, 
        growth_rate=[36,36,36,36],
        num_layers=[5,5,5,5],
        num_td2_blocks=[10,10,10,10],
        out_block=[5,5,5,5],
        block_comp=[0.5, 0.5, 0.5, 0.5],
        trans_comp_factor=[2, 2, 2, 1],
        kernel_size=3,
        is_cbr=True,
        dropout_p=0.2,
        relu_type='prelu',
        use_td3_block=True,
        use_multi_dilation=True,
        use_bottle_layer=True,  # Whether to use bottle layer in TD2 blocks
        ):
        super().__init__()
        num_blocks = len(num_td2_blocks)
        blocks = []
        current_channels = in_ch
        
        for i in range(num_blocks):
            if use_td3_block:
                block = TD3Block(
                    in_ch = current_channels,
                    growth_rate   = growth_rate[i],
                    num_layers    = num_layers[i],
                    num_td2_blocks = num_td2_blocks[i],
                    out_block     = out_block[i],
                    block_comp    = block_comp[i],
                    trans_comp_factor = trans_comp_factor[i],
                    kernel_size   = kernel_size,
                    is_cbr = is_cbr,
                    dropout_p = dropout_p,
                    relu_type = relu_type,
                    use_multi_dilation=use_multi_dilation,
                    use_bottle_layer=use_bottle_layer,
                )
                current_channels = block.transition_out
            else:
                block_stack = []
                for _ in range(num_td2_blocks[i]):
                    block = TD2Block(
                        in_ch=current_channels,
                        growth_rate=growth_rate[i],
                        num_layers=num_layers[i],
                        out_block=out_block[i],
                        block_comp=block_comp[i],
                        kernel_size=kernel_size,
                        is_cbr=is_cbr,
                        dropout_p=dropout_p,
                        relu_type=relu_type,
                        use_multi_dilation=use_multi_dilation,
                        use_bottle_layer=use_bottle_layer,
                    )
                    block_stack.append(block)
                    current_channels = int(growth_rate[i] * out_block[i] * block_comp[i])
                block = nn.Sequential(*block_stack)
            
            blocks.append(block)

        self.net = nn.Sequential(*blocks)
        self.in_ch = in_ch
        self.out_ch = current_channels
        
    def forward(self, x, lengths = None):
        out = self.net(x) 
        if lengths is not None :
            out = torch.stack([torch.mean(out[index][:, 0:i], 1) for index, i in enumerate(lengths)],0)
        else :
            out = reduce(out, 'b c t -> b c', reduction='mean')
        
        return out

class TD3Block(nn.Module):
    def __init__(
        self,
        in_ch,
        growth_rate,
        num_layers,
        num_td2_blocks,
        out_block,
        block_comp,
        trans_comp_factor,
        kernel_size,
        is_cbr,
        dropout_p,
        relu_type,
        use_multi_dilation=True,
        use_bottle_layer=True,  # Whether to use bottle layer in TD2 blocks
        ):
        super().__init__()
        self.num_td2_blocks = num_td2_blocks
        td2_block = []
        for i in range(num_td2_blocks):
            td2_block.append(
                TD2Block(
                    in_ch = in_ch,
                    growth_rate = growth_rate,
                    num_layers  = num_layers,
                    out_block   = out_block,
                    block_comp  = block_comp,
                    kernel_size = kernel_size,
                    is_cbr = is_cbr,
                    dropout_p = dropout_p,
                    relu_type = relu_type,
                    use_multi_dilation=use_multi_dilation,
                    use_bottle_layer=use_bottle_layer,
                )
            )
            in_ch += int(growth_rate * out_block * block_comp)

        self.td2_block = nn.ModuleList(td2_block)

        self.transition = BasicConv(
                            in_channels  = in_ch,
                            out_channels = in_ch // trans_comp_factor,
                            kernel_size  = 1,
                            is_cbr = is_cbr,
                            dropout_p = dropout_p,
                            relu_type = relu_type,)

        self.transition_out = self.transition.out_ch

    def forward(self, x) :
        out = x
        for i in range(self.num_td2_blocks):
            block_out = self.td2_block[i](out)
            out = torch.cat([out, block_out], dim = 1)
        out = self.transition(out)

        return out

class TD2Block(nn.Module):
    def __init__(
        self,
        in_ch,
        growth_rate,
        num_layers,
        out_block,
        block_comp,
        kernel_size,
        is_cbr,
        dropout_p,
        relu_type,
        use_multi_dilation=True,  # True: Multi_Dilated_Dense_Block + BasicConv, False: DenseBlock + DBasicBlock
        use_bottle_layer=True,    # Whether to use bottle layer for channel reduction
    ):
        super().__init__()
        self.bc_ch = growth_rate * 4
        self.in_ch = in_ch
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.use_multi_dilation = use_multi_dilation
        self.use_bottle_layer = use_bottle_layer

        if use_bottle_layer and self.bc_ch < in_ch:
            self.bottle_layer = BasicConv(
                in_channels=in_ch,
                out_channels=self.bc_ch,
                kernel_size=1,
                is_cbr=is_cbr,
                dropout_p=dropout_p,
                relu_type=relu_type,
            )
        else:
            self.bc_ch = in_ch

        if use_multi_dilation:
            self.init_layer = BasicConv(
                in_channels=self.bc_ch,
                out_channels=growth_rate * num_layers,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                is_cbr=is_cbr,
                dropout_p=dropout_p,
                relu_type=relu_type,
            )
            
            self.MDDB = Multi_Dilated_Dense_Block(
                growth_rate=growth_rate,
                num_layers=num_layers,
                kernel_size=kernel_size,
                is_cbr=is_cbr,
                dropout_p=dropout_p,
                relu_type=relu_type,
            )
            self.reduction_in_channels = growth_rate * num_layers
        else:
            self.init_layer = DBasicBlock(
                in_channels=self.bc_ch,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dilation=1,
                is_cbr=is_cbr,
                dropout_p=dropout_p,
                relu_type=relu_type,
            )
            
            self.MDDB = DenseBlock(
                num_layer=num_layers-1,
                in_channels=self.bc_ch + growth_rate,
                growth_rate=growth_rate,
                block=DBasicBlock,
                kernel_size=kernel_size,
                dilation=2,
                is_cbr=is_cbr,
                dropout_p=dropout_p,
                relu_type=relu_type,
            )
         
            self.reduction_in_channels = self.bc_ch + growth_rate * num_layers

        self.block_reduction_layer = BasicConv(
            in_channels=self.reduction_in_channels,
            out_channels=int(growth_rate * out_block * block_comp),
            kernel_size=1,
            is_cbr=is_cbr,
            dropout_p=dropout_p,
            relu_type=relu_type,
        )

    def forward(self, x):
        out = x
        if self.use_bottle_layer and self.bc_ch < self.in_ch:
            out = self.bottle_layer(out)
        out = self.init_layer(out)
        out = self.MDDB(out)
        out = self.block_reduction_layer(out)
        return out

class Multi_Dilated_Dense_Block(nn.Module):
    def __init__(
        self,
        growth_rate,
        num_layers,
        kernel_size,
        is_cbr,
        dropout_p,
        relu_type,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        layers = []
        for i in range(num_layers - 1):
            d = int(2**(i+1))
            layers.append(BasicConv(
                in_channels=growth_rate,
                out_channels=growth_rate * (num_layers-i-1),
                kernel_size=kernel_size,
                dilation=d,
                padding=(kernel_size//2) * d,
                is_cbr=is_cbr,
                dropout_p=dropout_p,
                relu_type=relu_type,
            ))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        lst = []
        for i in range(self.num_layers):
            lst.append(x[:, i*self.growth_rate:(i+1)*self.growth_rate])
        
        def update(inp_, n):
            for j in range(self.num_layers-n-1):
                lst[j+1+n] += inp_[:, j*self.growth_rate:(j+1)*self.growth_rate]

        for i, layer in enumerate(self.layers):
            update(layer(lst[i]), i)

        out = torch.cat([*lst], dim=1)
        return out 