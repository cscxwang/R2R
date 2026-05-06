import torch
import torch.nn as nn
import torch.nn.functional as F
from net.feature_bank_5D import DegradationMemory
import matplotlib.pyplot as plt
import os
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

'''
ref.
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''
class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        self = self.cuda()
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size).cuda()
        with torch.no_grad():
            self.forward(imgs, interact_label=[0])
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFKeyEncoder(nn.Module):
    def __init__(self, width=32, middle_blk_num=1, enc_blk_nums=[1,1,1,28]):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.middle_blks = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

    def forward(self, x):
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)

        return x, encs

class Classifier(nn.Module):
    def __init__(self, in_channels=512, num_classes=5):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 输出 [b, 512, 1, 1]

        self.fc = nn.Sequential(
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)       # [b, 512, 1, 1]
        x = torch.flatten(x, 1)   # [b, 512]
        x = self.fc(x)            # [b, num_classes]
        return x                  # 直接输出 logits，后面配合 CrossEntropyLoss
class NAFMemEncoder(nn.Module):
    def __init__(self, image_channels=3, deg_channel=1, width=32, middle_blk_num=1, enc_blk_nums=[2,2,4,8], num_classes=5):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=image_channels+deg_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.middle_blks = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        self.classify = Classifier(in_channels=chan//2, num_classes=num_classes)

    def forward(self, x):
        encs = []
        x = self.intro(x)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            x = down(x)
            encs.append(x)
        y = self.middle_blks(x)

        prob = self.classify(F.relu(encs[-2]))

        return y, prob

class NAFDecoder(nn.Module):
    def __init__(self, chan=512, dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

    def forward(self, x, encs, readout):
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip

            x = decoder(x)
        return x

class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)

class ValueProjection(nn.Module):
    def __init__(self, indim, valuedim):
        super().__init__()
        self.value_proj = nn.Conv2d(indim, valuedim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.value_proj.weight.data)
        nn.init.zeros_(self.value_proj.bias.data)

    def forward(self, x):
        return self.value_proj(x)

class ValueComp(nn.Module):
    def __init__(self, indim, valuedim):
        super().__init__()
        self.value_proj = nn.Conv2d(indim, valuedim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.value_proj.weight.data)
        nn.init.zeros_(self.value_proj.bias.data)

    def forward(self, x):
        return self.value_proj(x)

class skipP(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.channel = channel
        self.fuse_gate = nn.Conv2d(channel*2, channel, kernel_size=1, groups=channel)

    def forward(self, x, readout):

        B, C, H, W = x.shape

        readout = F.interpolate(readout, scale_factor=2, mode="bilinear", align_corners=True)
        readout = self.sel_channel(readout, self.channel)

        interleaved = torch.zeros(B, 2 * x.shape[1], x.shape[2], x.shape[3]).to(x.device)
        interleaved[:, 0::2, :, :] = x  # 偶数索引：特征A
        interleaved[:, 1::2, :, :] = readout
        gate = torch.sigmoid(self.fuse_gate(interleaved))
        x = x * (1 - gate) + readout * gate
        return x

    def sel_channel(self, features, n):
        B, C, H, W = features.shape

        weights = features.mean(dim=(2, 3))  # [B, C]

        topk_values, topk_indices = torch.topk(weights, k=n, dim=1)  # [B, n]

        batch_indices = torch.arange(B).unsqueeze(1).to(features.device)  # [B, 1]
        selected_features = features[batch_indices, topk_indices]  # [B, n, H, W]
        return selected_features
class R2R(nn.Module):
    def __init__(self, opt=None,  img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], key_dim=64, value_dim=512, is_train=True, stage=0, num_classes=5, train_mode="pretrain"):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = NAFKeyEncoder(width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums)
        self.mem_encoder = NAFMemEncoder(img_channel,  deg_channel=1, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=[2, 2, 4, 8], num_classes=self.num_classes)
        self.decoder = NAFDecoder(chan=width*16, dec_blk_nums=dec_blk_nums)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.padder_size = 2 ** len(enc_blk_nums)

        self.key_proj = KeyProjection(512, keydim=key_dim)

        self.key_comp = nn.Conv2d(512, value_dim, kernel_size=3, padding=1)

        self.value_proj = ValueProjection(512, valuedim=key_dim)

        self.value_comp = ValueComp(512, valuedim=value_dim)

        self.dm = DegradationMemory(opt, T_max=64, key_dim=key_dim, value_dim=value_dim)

        self.is_train = is_train
        self.stage = stage
        self.train_mode = train_mode

        self.fuse_gate = nn.Conv2d(2 * value_dim, value_dim, kernel_size=1, groups=value_dim)

    def forward(self, inp, clean=None, ids=None, interact_label=None):
        inp = self.check_image_size(inp)
        if self.is_train and self.train_mode == "pretrain":
            inp, inp_ = inp.chunk(2, dim=0)
        B, C, H, W = inp.shape
        x = self.intro(inp)
        x, encs = self.encoder(x)

        qk = self.key_proj(x)
        qv = self.key_comp(x)

        if self.is_train and self.train_mode == "pretrain":
            self.dm.clear_grad(self.stage)
            clean = self.check_image_size(clean)
            _, clean_ = clean.chunk(2, dim=0)
            y =  inp_[:, :, :H, :W] - clean_[:, :, :H, :W]
            y = torch.mean(y, dim=1, keepdim=True)
            mem_frame = torch.cat([clean_[:, :, :H, :W],y], dim=1)
            mem_feat, prob = self.mem_encoder(mem_frame)
            mk = self.value_proj(mem_feat)
            mv = self.value_comp(mem_feat)
            gt_ids = []
            for id in ids:
                id = id - 2
                if id <= 0:
                    gt_ids.append(0)
                else:
                    gt_ids.append(id.item())
            gt_ids = torch.tensor(gt_ids, device=ids.device)
            gt_ids__, gt_ids_ =  gt_ids.chunk(2, dim=0)

            self.dm.update_bank("denoise", mk[[i for i, id_val in enumerate(gt_ids_) if id_val == 0]], mv[[i for i, id_val in enumerate(gt_ids_) if id_val == 0]])
            self.dm.update_bank("derain", mk[[i for i, id_val in enumerate(gt_ids_) if id_val == 1]], mv[[i for i, id_val in enumerate(gt_ids_) if id_val == 1]])
            self.dm.update_bank("dehaze", mk[[i for i, id_val in enumerate(gt_ids_) if id_val == 2]], mv[[i for i, id_val in enumerate(gt_ids_) if id_val == 2]])
            self.dm.update_bank("deblur", mk[[i for i, id_val in enumerate(gt_ids_) if id_val == 3]], mv[[i for i, id_val in enumerate(gt_ids_) if id_val == 3]])
            self.dm.update_bank("lowlight", mk[[i for i, id_val in enumerate(gt_ids_) if id_val == 4]], mv[[i for i, id_val in enumerate(gt_ids_) if id_val == 4]])

            scores, read_out = self.dm.get_deg_prompt(qk, interact_label=gt_ids__)
        else:
            if self.is_train:
                gt_ids = []
                for id in ids:
                    id = id - 2
                    if id <= 0:
                        gt_ids.append(0)
                    else:
                        gt_ids.append(id.item())
                gt_ids = torch.tensor(gt_ids, device=ids.device)
                interact_label = gt_ids
            qk_ = F.interpolate(qk, size=(8, 8), mode="area")
            scores, read_out = self.dm.get_deg_prompt(qk_, interact_label=interact_label)
            read_out = F.interpolate(read_out, size=(qv.shape[-2], qv.shape[-1]), mode='area')

        if self.is_train and scores is None:
            scores = torch.ones((qk.shape[0], self.num_classes)).cuda()
            read_out = torch.zeros_like(qv).cuda()

        interleaved = torch.zeros(B, 2 * x.shape[1], x.shape[2], x.shape[3]).to(x.device)
        interleaved[:, 0::2, :, :] = qv  # 偶数索引：特征A
        interleaved[:, 1::2, :, :] = read_out
        gate = torch.sigmoid(self.fuse_gate(interleaved))
        x = qv * (1 - gate) + read_out * gate

        x = self.decoder(x, encs, read_out)
        x = self.ending(x)
        x = x + inp
        x_restore = x[:, :, :H, :W]

        if self.is_train and self.train_mode == "pretrain":
            return x_restore, prob, gt_ids, scores
        if self.is_train:
            return x_restore, scores, gt_ids

        return x_restore

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class R2RTest(nn.Module):
    def __init__(self, ckpt_path, prompts_path, prompts_name, train_mode="finetune"):
        super().__init__()
        net = R2R(opt=None, is_train=False, train_mode=train_mode)
        state_dict = torch.load(ckpt_path+prompts_name+".ckpt")['state_dict']
        state = {}
        for key in state_dict.keys():
            state[key[4:]] = state_dict[key]
        net.load_state_dict(state, strict=True)
        net.dm.load_prompts(prompts_name=prompts_name, save_root=prompts_path)
        self.net = net
    def forward(self, x, interact_label=None):
        return self.net(x, interact_label=interact_label)

class R2RLocal(Local_Base, R2RTest):
    def __init__(self, train_size=(1, 3, 224, 224), fast_imp=False, ckpt_path="", prompts_path="", prompts_name="last", train_mode="finetune"):
        Local_Base.__init__(self)
        R2RTest.__init__(self, ckpt_path, prompts_path, prompts_name, train_mode=train_mode)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
