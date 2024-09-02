import torch
import torch.nn as nn

from yucca.modules.networks.networks.YuccaNet import YuccaNet
from models.conv_blocks.blocks import DoubleConvDropoutNormNonlin, MultiLayerConvDropoutNormNonlin


class UNet(YuccaNet):
    def __init__(
        self,
        reconstruction: bool = False,
        prediction: bool = False,
        input_channels: int = 1,
        output_channels: int = 1,
        starting_filters: int = 64,
        encoder_block: nn.Module = MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        decoder_block: nn.Module = MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        use_skip_connections: bool = False,
        deep_supervision: bool = False,
    ):
        super().__init__()

        if not (reconstruction or prediction):
            print("Instantiating Unet in headless mode.")

        self.encoder_block = encoder_block
        self.decoder_block = decoder_block

        self.encoder = UNetEncoder(input_channels=input_channels, starting_filters=starting_filters, basic_block=encoder_block)
        # dim = starting_filters * 16
        self.num_classes = output_channels

        if reconstruction:
            self.rec_head = UNetDecoder(
                output_channels=output_channels,
                use_skip_connections=use_skip_connections,
                basic_block=decoder_block,
                starting_filters=starting_filters,
            )

        if prediction:
            self.pred_head = UNetDecoder(
                output_channels=output_channels,
                use_skip_connections=True,
                deep_supervision=deep_supervision,
                basic_block=decoder_block,
                starting_filters=starting_filters,
            )

        self.reconstruction = reconstruction
        self.prediction = prediction

    def forward(self, x):
        enc = self.encoder(x)

        if self.prediction:
            return self.pred_head(enc)
        elif self.reconstruction:
            return self.rec_head(enc)
        else:  # headless mode
            return enc


class UNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        starting_filters: int = 64,
        conv_op=nn.Conv3d,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True},
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        weightInitializer=None,
        basic_block=DoubleConvDropoutNormNonlin,
    ) -> None:
        super().__init__()

        # Task specific
        self.filters = starting_filters

        # Model parameters
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.basic_block = basic_block

        self.pool_op = nn.MaxPool3d

        self.in_conv = self.basic_block(
            input_channels=input_channels,
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool1 = self.pool_op(2)
        self.encoder_conv1 = self.basic_block(
            input_channels=self.filters,
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool2 = self.pool_op(2)
        self.encoder_conv2 = self.basic_block(
            input_channels=self.filters * 2,
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool3 = self.pool_op(2)
        self.encoder_conv3 = self.basic_block(
            input_channels=self.filters * 4,
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool4 = self.pool_op(2)
        self.encoder_conv4 = self.basic_block(
            input_channels=self.filters * 8,
            output_channels=self.filters * 16,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.pool1(x0)
        x1 = self.encoder_conv1(x1)

        x2 = self.pool2(x1)
        x2 = self.encoder_conv2(x2)

        x3 = self.pool3(x2)
        x3 = self.encoder_conv3(x3)

        x4 = self.pool4(x3)
        x4 = self.encoder_conv4(x4)

        return [x0, x1, x2, x3, x4]


class UNetDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 64,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        dropout_in_decoder=False,
        weightInitializer=None,
        basic_block=DoubleConvDropoutNormNonlin,
        deep_supervision=False,
        use_skip_connections=True,
    ) -> None:
        super().__init__()

        # Task specific
        self.num_classes = output_channels
        self.filters = starting_filters

        # Model parameters
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.basic_block = basic_block
        self.deep_supervision = deep_supervision
        self.use_skip_connections = use_skip_connections

        self.upsample = torch.nn.ConvTranspose3d

        # Decoder
        if not dropout_in_decoder:
            old_dropout_p = self.dropout_op_kwargs["p"]
            self.dropout_op_kwargs["p"] = 0.0

        self.upsample1 = self.upsample(self.filters * 16, self.filters * 8, kernel_size=2, stride=2)
        self.decoder_conv1 = self.basic_block(
            input_channels=self.filters * (16 if self.use_skip_connections else 8),
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample2 = self.upsample(self.filters * 8, self.filters * 4, kernel_size=2, stride=2)
        self.decoder_conv2 = self.basic_block(
            input_channels=self.filters * (8 if self.use_skip_connections else 4),
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample3 = self.upsample(self.filters * 4, self.filters * 2, kernel_size=2, stride=2)
        self.decoder_conv3 = self.basic_block(
            input_channels=self.filters * (4 if self.use_skip_connections else 2),
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample4 = self.upsample(self.filters * 2, self.filters, kernel_size=2, stride=2)
        self.decoder_conv4 = self.basic_block(
            input_channels=self.filters * (2 if self.use_skip_connections else 1),
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.out_conv = self.conv_op(self.filters, self.num_classes, kernel_size=1)

        if self.deep_supervision:
            self.ds_out_conv0 = self.conv_op(self.filters * 16, self.num_classes, kernel_size=1)
            self.ds_out_conv1 = self.conv_op(self.filters * 8, self.num_classes, kernel_size=1)
            self.ds_out_conv2 = self.conv_op(self.filters * 4, self.num_classes, kernel_size=1)
            self.ds_out_conv3 = self.conv_op(self.filters * 2, self.num_classes, kernel_size=1)

        if not dropout_in_decoder:
            self.dropout_op_kwargs["p"] = old_dropout_p

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, xs):
        # We assume xs contains 5 elements. One for each of the skip connections and the bottleneck representation
        # The contents of xs is: [first skip connection, ..., last skip connection, bottleneck]
        assert isinstance(xs, list), type(xs)
        assert len(xs) == 5

        x_enc = xs[4]

        if self.use_skip_connections:
            x5 = torch.cat([self.upsample1(x_enc), xs[3]], dim=1)
            x5 = self.decoder_conv1(x5)

            x6 = torch.cat([self.upsample2(x5), xs[2]], dim=1)
            x6 = self.decoder_conv2(x6)

            x7 = torch.cat([self.upsample3(x6), xs[1]], dim=1)
            x7 = self.decoder_conv3(x7)

            x8 = torch.cat([self.upsample4(x7), xs[0]], dim=1)
            x8 = self.decoder_conv4(x8)
        else:
            x5 = self.decoder_conv1(self.upsample1(x_enc))
            x6 = self.decoder_conv2(self.upsample2(x5))
            x7 = self.decoder_conv3(self.upsample3(x6))
            x8 = self.decoder_conv4(self.upsample4(x7))

        # We only want to do multiple outputs during training, therefore it is only enabled
        # when grad is also enabled because that means we're training. And if for some reason
        # grad is enabled and you're not training, then there's other, bigger problems.
        if self.deep_supervision and torch.is_grad_enabled():
            ds0 = self.ds_out_conv0(xs[4])
            ds1 = self.ds_out_conv1(x5)
            ds2 = self.ds_out_conv2(x6)
            ds3 = self.ds_out_conv3(x7)
            ds4 = self.out_conv(x8)
            return [ds4, ds3, ds2, ds1, ds0]

        logits = self.out_conv(x8)

        return logits


def unet_b(
    reconstruction: bool = False,
    prediction: bool = True,
    input_channels: int = 1,
    output_channels: int = 1,
):
    return UNet(
        reconstruction=reconstruction,
        prediction=prediction,
        input_channels=input_channels,
        output_channels=output_channels,
        use_skip_connections=True,
        starting_filters=32,
    )


def unet_b_lw_dec(
    reconstruction: bool = False,
    prediction: bool = True,
    input_channels: int = 1,
    output_channels: int = 1,
):
    unet_model = UNet(
        reconstruction=reconstruction,
        prediction=prediction,
        input_channels=input_channels,
        output_channels=output_channels,
        decoder_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        use_skip_connections=False,
        starting_filters=32,
    )

    return unet_model


def unet_xl(
    reconstruction: bool = False,
    prediction: bool = True,
    input_channels: int = 1,
    output_channels: int = 1,
):
    return UNet(
        reconstruction=reconstruction,
        prediction=prediction,
        input_channels=input_channels,
        output_channels=output_channels,
        use_skip_connections=True,
    )


def light_weight_decoder(
    output_channels: int = 1,
    use_skip_connections: bool = False,
    starting_filters: int = 64,
):
    decoder_block = MultiLayerConvDropoutNormNonlin.get_block_constructor(1)
    return UNetDecoder(
        output_channels=output_channels,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        basic_block=decoder_block,
    )


def standard_decoder(
    output_channels: int = 1,
    use_skip_connections: bool = False,
    starting_filters: int = 64,
    deep_supervision: bool = False,
):
    decoder_block = MultiLayerConvDropoutNormNonlin.get_block_constructor(2)
    return UNetDecoder(
        output_channels=output_channels,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        basic_block=decoder_block,
        deep_supervision=deep_supervision,
    )


def unet_xl_lw_dec(
    reconstruction: bool = False,
    prediction: bool = True,
    input_channels: int = 1,
    output_channels: int = 1,
):
    unet_model = UNet(
        reconstruction=reconstruction,
        prediction=prediction,
        input_channels=input_channels,
        output_channels=output_channels,
        decoder_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        use_skip_connections=False,
    )

    return unet_model
