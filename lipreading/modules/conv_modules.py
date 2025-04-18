import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, is_cbr=True, dropout_p = False, relu_type = 'relu'):
        super().__init__()
        self.is_cbr = is_cbr
        self.out_ch = out_channels
        if is_cbr :
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.bn   = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
            if relu :
                if relu_type == 'relu' : self.relu = nn.ReLU()
                else : self.relu = nn.PReLU(num_parameters=out_channels) 
            else : None
        else :
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.bn   = nn.BatchNorm1d(in_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
            if relu :
                if relu_type == 'relu' : self.relu = nn.ReLU()
                else : self.relu = nn.PReLU(num_parameters=in_channels) 
            else : None

        self.drop = nn.Dropout(dropout_p) if dropout_p else None

    def forward(self, x):
        # x need to [B,C,T]
        if self.is_cbr : 
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            if self.drop is not None:
                x = self.drop(x)
        # brc    
        else :
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            x = self.conv(x)
            if self.drop is not None:
                x = self.drop(x)
        return x

class DBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, is_cbr, dropout_p, relu_type):
        super(DBasicBlock, self).__init__()
        self.cbr = BasicConv(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    dilation     = dilation,
                    padding      = (kernel_size//2) * dilation,
                    is_cbr       = is_cbr,
                    dropout_p    = dropout_p,
                    relu_type    = relu_type,
                    )
        
    def forward(self, x):
        out = self.cbr(x)
        return torch.cat([x, out], 1)

    

class DenseBlock(nn.Module):
    def __init__(self, num_layer, in_channels, growth_rate, block, kernel_size, dilation, is_cbr, dropout_p, relu_type):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, growth_rate, num_layer,  kernel_size, dilation, is_cbr, dropout_p, relu_type)
    def _make_layer(self, block, in_channels, growth_rate, num_layer, kernel_size, dilation, is_cbr, dropout_p, relu_type):
        layers = []
        for i in range(num_layer):
            d = dilation ** (i + 1)
            layers.append(block(in_channels+i*growth_rate, growth_rate, kernel_size, d, is_cbr, dropout_p, relu_type))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

