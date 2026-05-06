import os
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataset_utils import PromptTrainDataset
from net.model_3D import R2R
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options.options_3D import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
np.random.seed(1415926)
torch.manual_seed(1415926)

class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion='l2', reduction='mean'):
        super(EdgeLoss, self).__init__()

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).cuda()

        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.criterion(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

def clamp_restored(a, b):
    a = torch.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
    b_min = b.amin(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
    b_max = b.amax(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
    print(b_min)
    print(b_max)
    assert 1 == 0
    a_clamped = torch.max(torch.min(a, b_max), b_min)
    return a_clamped

def sobel_gradient(x):
    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=x.dtype, device=x.device)
    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]]]], dtype=x.dtype, device=x.device)
    grad_x = F.conv2d(x, sobel_x.expand(x.shape[1], 1, 3, 3), padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, sobel_y.expand(x.shape[1], 1, 3, 3), padding=1, groups=x.shape[1])
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

class R2RModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = R2R(opt, train_mode=opt.train_mode)
        self.loss_fn = nn.L1Loss()
        self.classify = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.psnr = PSNRLoss().eval()
        self.total_loss = 0.0
        self.count = 0

        self.total_kv_loss = 0.0

        self.total_psnr = 0.0
        self.total_acc = 0.0
        self.total_deg_acc = 0.0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        net_out = self.net(degrad_patch, clean_patch, de_id)
        if opt.train_mode == "pretrain":
            restored, prob, gt_ids, scores = net_out
            clean_target, _ = clean_patch.chunk(2, dim=0)
            gt_ids1, gt_ids2 = gt_ids.chunk(2, dim=0)
            classify_loss = self.classify(prob, gt_ids2) * 0.1
            deg_loss = self.classify(scores, gt_ids1) * 0.1
        else:
            restored, scores, gt_ids = net_out
            clean_target = clean_patch
            classify_loss = torch.tensor(0.0, device=restored.device)
            deg_loss = self.classify(scores, gt_ids) * 0.1
        restored = torch.clamp(restored, 0, 1)

        loss = self.loss_fn(restored, clean_target)

        label_fft3 = torch.fft.fft2(clean_target, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(restored, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
        loss_fft = self.loss_fn(pred_fft3, label_fft3) * 0.125

        loss = loss + classify_loss + deg_loss + loss_fft

        with torch.no_grad():
            PSNR = -self.psnr(restored, clean_target)
            self.total_psnr += PSNR
            deg_target = gt_ids1 if opt.train_mode == "pretrain" else gt_ids
            correct = (torch.argmax(scores, dim=1) == deg_target).sum().item() / deg_target.size(0)
            self.total_deg_acc += correct

            if opt.train_mode == "pretrain":
                acc = (torch.argmax(prob, dim=1) == gt_ids2).sum().item() / gt_ids2.size(0)
                self.total_acc += acc

        self.total_loss += loss.item()
        self.count += 1
        self.log("batch_loss", loss, prog_bar=True, logger=True)
        self.log("epoch_avg_train_loss", self.total_loss / self.count, prog_bar=True, logger=True)
        if opt.train_mode == "pretrain":
            self.log("epoch_avg_classify_acc", self.total_acc / self.count, prog_bar=True, logger=True)
        self.log("epoch_avg_deg_acc", self.total_deg_acc / self.count, prog_bar=True, logger=True)
        self.log("epoch_avg_PSNR", self.total_psnr / self.count, prog_bar=True, logger=True)

        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]

        self.log("lr", lr, logger=True)

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=opt.lr)
        if opt.train_mode == "pretrain":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=opt.warmup_epochs,
                                                      max_epochs=opt.epochs + 30, warmup_start_lr=1e-7)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-8)

        return [optimizer], [scheduler]

from lightning.pytorch.callbacks import TQDMProgressBar

class MyProgressBar(TQDMProgressBar):
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if opt.train_mode == "pretrain":
            with torch.no_grad():
                if trainer.current_epoch >= opt.epochs // 2:
                    pl_module.net.stage = 1
                else:
                    pl_module.net.dm.clear_grad()

        pl_module.total_loss = 0.0
        pl_module.count = 0
        pl_module.total_kv_loss = 0.0
        pl_module.total_acc = 0.0
        pl_module.total_deg_acc = 0.0
        pl_module.total_psnr = 0.0
        print("")  # 换行

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        if opt.train_mode == "pretrain":
            pl_module.net.dm.save_prompts(epoch="last", save_root=opt.prompt_dir)

def main():
    print("Options")
    print(opt)

    logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, filename="last", save_last=True, save_top_k=0)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    model = R2RModel()
    if opt.train_mode == "finetune":
        model.net.dm.load_prompts(prompts_name=opt.init_prompt_name, save_root=opt.init_prompt_dir)
        base_ckpt_path = os.path.join(opt.init_ckpt_dir, f"{opt.init_prompt_name}.ckpt")
        model.load_state_dict(torch.load(base_ckpt_path)["state_dict"], strict=True)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback, MyProgressBar()],
        gradient_clip_val=0.01,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
    )

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    main()
