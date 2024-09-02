import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.depthconv = conv_op(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.pointwiseconv = conv_op(input_channels, output_channels, kernel_size=1)
