from typing import Optional, Union
import torch.nn as nn
from models.networks.unet import light_weight_decoder, standard_decoder
from yucca.modules.networks.blocks_and_layers.conv_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    OutBlock,
)
from yucca.modules.networks.networks.YuccaNet import YuccaNet


class MedNeXt(YuccaNet):
    """
    From the paper: https://arxiv.org/pdf/2303.09975.pdf
    code source: https://github.com/MIC-DKFZ/MedNeXt/tree/main
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        contrastive: bool = False,
        rotation: bool = False,
        reconstruction: bool = False,
        prediction: bool = False,
        conv_op=nn.Conv3d,
        starting_filters: int = 32,
        enc_exp_r: Union[int, list] = 2,
        dec_exp_r: Union[int, list] = 2,
        kernel_size: int = 5,
        do_res: bool = True,
        do_res_up_down: bool = True,
        enc_block_counts: list = [2, 2, 2, 2, 2],
        dec_block_counts: list = [2, 2, 2, 2],
        norm_type="group",
        grn=False,
    ):
        super().__init__()
        self.contrastive = contrastive
        self.rotation = rotation
        self.reconstruction = reconstruction
        self.prediction = prediction
        print(
            f"Loaded MedNeXt with Contrastive: {self.contrastive}, Rotation: {self.rotation}, Reconstruction: {self.reconstruction}, Prediction: {self.prediction}"
        )

        dim = starting_filters * 16

        self.num_classes = output_channels

        self.encoder = MedNeXtEncoder(
            input_channels=input_channels,
            conv_op=conv_op,
            starting_filters=starting_filters,
            kernel_size=kernel_size,
            exp_r=enc_exp_r,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=enc_block_counts,
            norm_type=norm_type,
            grn=grn,
        )
        if self.contrastive:
            self.con_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(dim, 512))

        if self.rotation:
            self.rot_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(dim, 4), nn.Softmax(dim=1))

        # We dont use the mednext decoder during pretraining. Instantiate it here if you need it.
        self.rec_head = None
        self.pred_head = MedNeXtDecoder(
            output_channels=output_channels,
            starting_filters=starting_filters,
            exp_r=dec_exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=dec_block_counts,
            norm_type=norm_type,
            grn=grn,
        )

    def forward(self, x):
        assert self.rec_head is not None or self.pred_head is not None

        enc = self.encoder(x)
        if self.prediction:
            return self.pred_head(enc)

        y_hat_rot = self.rot_head(enc[4]) if self.rotation else None
        y_hat_con = self.con_head(enc[4]) if self.contrastive else None
        y_hat_rec = self.rec_head(enc) if self.reconstruction else None

        return y_hat_con, y_hat_rec, y_hat_rot


class MedNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_op=nn.Conv3d,
        starting_filters: int = 32,
        exp_r: Union[int, list] = [3, 4, 8, 8, 8],
        kernel_size: int = 5,
        do_res: bool = True,
        do_res_up_down: bool = True,
        block_counts: list = [3, 4, 8, 8, 8],
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        dim = "3d"

        self.stem = conv_op(input_channels, starting_filters, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[0],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[1],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[1],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[2],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[2],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[3],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[3],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * starting_filters,
            out_channels=16 * starting_filters,
            exp_r=exp_r[4],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 16,
                    out_channels=starting_filters * 16,
                    exp_r=exp_r[4],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[4])
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)
        return [x_res_0, x_res_1, x_res_2, x_res_3, x]


class MedNeXtDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 32,
        exp_r: Union[int, list] = [3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size: int = 5,
        dec_kernel_size: Optional[int] = None,
        deep_supervision: bool = False,
        do_res: bool = True,
        do_res_up_down: bool = True,
        block_counts: list = [8, 8, 4, 3],
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.output_channels = output_channels
        if kernel_size is not None:
            dec_kernel_size = kernel_size

        dim = "3d"

        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[0],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[0],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[1],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[1],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[2],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[2],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * starting_filters,
            out_channels=starting_filters,
            exp_r=exp_r[3],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[3],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.out_0 = OutBlock(in_channels=starting_filters, n_classes=self.output_channels, dim=dim)

        if self.deep_supervision:
            raise NotImplementedError

        self.block_counts = block_counts

    def forward(self, xs: list):
        # unpack the output of the encoder
        x_res_0, x_res_1, x_res_2, x_res_3, x = xs

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)

        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)

        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        return x


class MedNeXtDecoderSSL(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 32,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
        kernel_size: int = 5,  # Ofcourse can test kernel_size
        dec_kernel_size: Optional[int] = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = True,  # Can be used to individually test residual connection
        do_res_up_down: bool = True,  # Additional 'res' connection on up and down convs
        block_counts: list = [3, 4, 8, 8, 8, 8, 8, 4, 3],  # Can be used to test staging ratio:
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.output_channels = output_channels

        if kernel_size is not None:
            dec_kernel_size = kernel_size

        dim = "3d"

        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[5])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[6])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[7])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * starting_filters,
            out_channels=starting_filters,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[8])
            ]
        )

        self.out_0 = OutBlock(in_channels=starting_filters, n_classes=self.output_channels, dim=dim)

        if self.deep_supervision:
            raise NotImplementedError

        self.block_counts = block_counts

    def forward(self, x: list):
        x = self.up_3(x)
        x = self.dec_block_3(x)

        x = self.up_2(x)
        x = self.dec_block_2(x)

        x = self.up_1(x)
        x = self.dec_block_1(x)

        x = self.up_0(x)
        x = self.dec_block_0(x)

        x = self.out_0(x)

        return x


def mednext_s3(
    input_channels: int,
    num_classes: int = 1,
    conv_op=nn.Conv3d,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = False,
    grn: bool = False,
    deep_supervision: bool = False,
):
    return MedNeXt(
        input_channels=input_channels,
        output_channels=num_classes,
        conv_op=conv_op,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
        kernel_size=3,
        enc_exp_r=2,
        dec_exp_r=2,
        enc_block_counts=[2, 2, 2, 2, 2],
        dec_block_counts=[2, 2, 2, 2],
        grn=grn,
        deep_supervision=deep_supervision,
    )


def mednext_s3_lw_dec(
    input_channels: int,
    num_classes: int = 1,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
):
    net = mednext_s3(
        input_channels=input_channels,
        num_classes=num_classes,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=False,
    )

    net.rec_head = light_weight_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)
    net.pred_head = None

    return net


def mednext_s3_std_dec(
    input_channels: int,
    num_classes: int = 1,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = True,
    deep_supervision: bool = True,
):
    net = mednext_s3(
        input_channels=input_channels,
        num_classes=num_classes,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
        deep_supervision=deep_supervision,
    )

    # sanity check
    assert (prediction and not reconstruction) or (reconstruction and not prediction)

    if reconstruction:
        assert not deep_supervision
        print("Using a standard unet decoder as reconstruction head")
        net.rec_head = standard_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)

    if prediction:
        print("Using a standard unet decoder as prediction head")
        net.pred_head = standard_decoder(
            output_channels=num_classes, starting_filters=32, use_skip_connections=True, deep_supervision=deep_supervision
        )

    return net


def mednext_m3(
    input_channels: int,
    num_classes: int = 1,
    conv_op=nn.Conv3d,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = False,
):
    return MedNeXt(
        input_channels=input_channels,
        output_channels=num_classes,
        conv_op=conv_op,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
        kernel_size=3,
        enc_exp_r=[2, 3, 4, 4, 4],
        dec_exp_r=[4, 4, 3, 2],
        enc_block_counts=[3, 4, 4, 4, 4],
        dec_block_counts=[4, 4, 4, 3],
    )


def mednext_m3_lw_dec(
    input_channels: int,
    num_classes: int = 1,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
):
    net = mednext_m3(
        input_channels=input_channels,
        num_classes=num_classes,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=False,
    )

    net.rec_head = light_weight_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)
    net.pred_head = None

    return net


def mednext_m3_std_dec(
    input_channels: int,
    num_classes: int = 1,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = True,
):
    net = mednext_m3(
        input_channels=input_channels,
        num_classes=num_classes,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
    )

    # sanity check
    assert (prediction and not reconstruction) or (reconstruction and not prediction)

    if reconstruction:
        print("Using a standard unet decoder as reconstruction head")
        net.rec_head = standard_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)

    if prediction:
        print("Using a standard unet decoder as prediction head")
        net.pred_head = standard_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=True)

    return net


def mednext_l3(
    input_channels: int,
    num_classes: int = 1,
    conv_op=nn.Conv3d,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = False,
):
    return MedNeXt(
        input_channels=input_channels,
        output_channels=num_classes,
        conv_op=conv_op,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
        kernel_size=3,
        enc_exp_r=[3, 4, 8, 8, 8],
        dec_exp_r=[8, 8, 4, 3],
        enc_block_counts=[3, 4, 8, 8, 8],
        dec_block_counts=[8, 8, 4, 3],
    )


def mednext_l3_lw_dec(
    input_channels: int,
    num_classes: int = 1,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
):
    net = mednext_l3(
        input_channels=input_channels,
        num_classes=num_classes,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=False,
    )

    net.rec_head = light_weight_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)
    net.pred_head = None

    return net


def mednext_l3_std_dec(
    input_channels: int,
    num_classes: int = 1,
    contrastive: bool = False,
    rotation: bool = False,
    reconstruction: bool = False,
    prediction: bool = True,
):
    net = mednext_l3(
        input_channels=input_channels,
        num_classes=num_classes,
        contrastive=contrastive,
        rotation=rotation,
        reconstruction=reconstruction,
        prediction=prediction,
    )

    # sanity check
    assert (prediction and not reconstruction) or (reconstruction and not prediction)

    if reconstruction:
        print("Using a standard unet decoder as reconstruction head")
        net.rec_head = standard_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=False)

    if prediction:
        print("Using a standard unet decoder as prediction head")
        net.pred_head = standard_decoder(output_channels=num_classes, starting_filters=32, use_skip_connections=True)

    return net
