import torch
from torch import nn as nn
from torch.nn import functional as F
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .spynet_arch import SpyNet

class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()
        # 마지막 빼고
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        # 처음 빼고
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        #1~14에 0~13의 정보 추가(앞 프레임의 정보)
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        #0~13에 1~14의 정보 추가(뒷 프레임의 정보)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        # 14~0
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            # x_i: [1, 3, 270, 480]
            # x_i: 14~0 frame
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                # feat_prop을 grid로 바꾸고 grid+flow하고 grid sampling으로 feat_prop sampling
                # feat_prop: [1, 64, 270, 480]
                # flow_warp: feature map에 flow를 추가
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            # feat_prop: [1, 67, 270, 480]
            feat_prop = self.backward_trunk(feat_prop)
            # feat_prop: [1, 64, 270, 480]
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            # backward feat_prop와 forward feat_prop fusion
            # 128 channel을 64 channel로 합침, 1x1 convolution으로
            out = self.lrelu(self.fusion(out))
            # 3x3 convolution, 64 channel-> 64x4 channel, pixel_shuffle(2)로 64x4 -> 64 channel
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            # 3x3 convolution, 64 channel-> 64x4 channel, pixel_shuffle(2)로 64x4 -> 64 channel
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            # 3x3 convolution, channel 64->3
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            # x_i를 4x
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            # base에 flow 정보 추가
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            # kernel size 3x3
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # output: [1, 64, h, w]
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)
