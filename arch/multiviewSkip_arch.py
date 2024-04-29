import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicvsr_arch import BasicVSR

class MultiViewSkipSR(nn.Module):
    def __init__(self, num_feat=64, num_block=15, load_path=None, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat
        # SR
        self.basicvsr = BasicVSR(num_feat, num_block, spynet_path)
        if load_path:
            self.basicvsr.load_state_dict(torch.load(load_path)['params'], strict=False)
        # 1x1 conv
        self.conv1x1 = nn.Conv2d(9, 3, 1, 1, 0)
        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        b, n, _, _, _ = x[0].size()
        num_view = len(x)
        x_lq = x[num_view//2]
        
        out_l = []
        view =  []
        for lq in x:
            view.append(self.basicvsr(lq))
        
        # view1 = self.basicvsr(x1)
        # view2 = self.basicvsr(x2)
        # view3 = self.basicvsr(x3)

        view_cat = torch.cat(view, 2)
        for i in range(0, n):
            x_i = view_cat[:, i, :, :, :]
            out = self.lrelu(self.conv1x1(x_i))
            out_l.extend(out)
        output = torch.stack(out_l, dim=0)
        output = torch.unsqueeze(output, 0)
        _, _, c, h, w = output.shape
        x = F.interpolate(x_lq, size=(c, h, w), mode='trilinear', align_corners=False)
        return output + x
