import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset, DeblurTestDataset, LOLTestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model_5D import R2R, R2RLocal
from utils.schedulers import *
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def patch_inference(img_lq, model, tile=None, tile_overlap=32, scale=1, interact_label=None):
    if tile is None:
        output = model(img_lq, interact_label=interact_label)
        if isinstance(output, list):
            output = output[-1]
    else:
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*scale, w*scale).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch, interact_label=interact_label)
                if isinstance(out_patch, list):
                    out_patch = out_patch[-1]
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
        output = E.div_(W)

    return output
class R2RModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = R2R(opt=None, is_train=False)
        self.loss_fn = nn.L1Loss()

    def forward(self, x, interact_label=None):
        return self.net(x, interact_label=interact_label)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]

def test_Denoise(net, dataset, sigma=15, interact_label=None):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    lpips = []
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored = patch_inference(degrad_patch, net,tile=None,  interact_label=[interact_label])
            restored = torch.clamp(restored, 0, 1)
            lpips.append(calc_lpips(clean_patch, restored).cpu().numpy())
            temuip_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + clean_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg, np.mean(lpips)

def test_Derain_Dehaze(net, dataset, task="derain", interact_label=None):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    lpips = []
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = patch_inference(degrad_patch, net, tile=None,  interact_label=[interact_label])

            restored = torch.clamp(restored, 0, 1)
            lpips.append(calc_lpips(clean_patch, restored).cpu().numpy())
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg, np.mean(lpips)

def test_Deblur(net, dataset, task="deblur", interact_label=None):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    lpips = []
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = patch_inference(degrad_patch, net, tile=None,  interact_label=[interact_label])
            restored = torch.clamp(restored, 0, 1)
            lpips.append(calc_lpips(clean_patch, restored).cpu().numpy())
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg, np.mean(lpips)

def test_Lowlight(net, dataset, task="lowlight", interact_label=None):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = patch_inference(degrad_patch, net, tile=None,  interact_label=[interact_label])
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for lowlight, 5 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="data/Test/Denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/Test/Derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/Test/Dehaze/", help='save path of test hazy images')
    parser.add_argument('--deblur_path', type=str, default="data/Test/Deblur/", help='save path of test blur images')
    parser.add_argument('--lowlight_path', type=str, default="data/Test/Lowlight/", help='save path of test lowlight images')
    parser.add_argument('--output_path', type=str, default="output/5D/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="train_ckpt_5D_finetune/", help='checkpoint save path')
    parser.add_argument('--prompt_dir', type=str, default="save_prompts_5D/", help='prompt save path')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_name

    denoise_splits = ["cbsd68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path, i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))

    name = "last"
    if testopt.mode == 3:
        net = R2RLocal(train_size=(1, 3, 224, 224), ckpt_path=ckpt_path, prompts_path=testopt.prompt_dir, prompts_name=name, train_mode="finetune").cuda()
    else:
        net = R2RLocal(ckpt_path=ckpt_path, prompts_path=testopt.prompt_dir, prompts_name=name, train_mode="finetune").cuda()
    testopt.derain_path = "data/Test/Derain/"
    testopt.dehaze_path = "data/Test/Dehaze/"
    testopt.denoise_path = "data/Test/Denoise/"
    testopt.deblur_path = "data/Test/Deblur/"
    testopt.lowlight_path = "data/Test/Lowlight/"

    net.eval()
    p = []
    s = []

    if testopt.mode == 0:
        for testset, name in zip(denoise_tests, denoise_splits):

            print('Start {} testing Sigma=25...'.format(name))
            _, _, lpips = test_Denoise(net, testset, sigma=25, interact_label=testopt.mode)
            print(lpips)

    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, task='derain', addnoise=False, sigma=15)
            _, _, lpips = test_Derain_Dehaze(net, derain_set, task="derain", interact_label=testopt.mode)
            print(lpips)

    elif testopt.mode == 2:
        print('Start testing SOTS...')
        dehaze_set = DerainDehazeDataset(testopt,task='dehaze', addnoise=False, sigma=15)
        test_Derain_Dehaze(net, dehaze_set, task="dehaze", interact_label=testopt.mode)

    elif testopt.mode == 3:
        print('Start testing GoPro...')
        deblur_set = DeblurTestDataset(testopt, addnoise=False, sigma=15)
        _, _, lpips = test_Deblur(net, deblur_set, task="deblur", interact_label=testopt.mode)
        print(lpips)
    elif testopt.mode == 4:
        print('Start testing LOL-v1...')
        lowlight_set = LOLTestDataset(testopt, addnoise=False, sigma=15)
        test_Lowlight(net, lowlight_set, task="lowlight", interact_label=testopt.mode)

    elif testopt.mode == 5:
        for testset, name in zip(denoise_tests, denoise_splits):

            print('Start {} testing Sigma=25...'.format(name))
            p1, s1, _ = test_Denoise(net, testset, sigma=25, interact_label=0)
            p.append(p1)
            s.append(s1)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt,task="derain",addnoise=False, sigma=15)
            p2, s2, _ = test_Derain_Dehaze(net, derain_set, task="derain", interact_label=1)

            p.append(p2)
            s.append(s2)

        print('Start testing SOTS...')
        dehaze_set = DerainDehazeDataset(testopt,task="dehaze",addnoise=False, sigma=15)
        p3, s3, _ = test_Derain_Dehaze(net, dehaze_set, task="dehaze", interact_label=2)
        p.append(p3)
        s.append(s3)

        print('Start testing GoPro...')
        deblur_set = DeblurTestDataset(testopt, addnoise=False, sigma=15)
        p4, s4, _ = test_Deblur(net, deblur_set, task="deblur", interact_label=3)
        p.append(p4)
        s.append(s4)

        print('Start testing LOL-v1...')
        lowlight_set = LOLTestDataset(testopt, addnoise=False, sigma=15)
        p5, s5 = test_Lowlight(net, lowlight_set, task="lowlight", interact_label=4)
        p.append(p5)
        s.append(s5)
        print(f"Avg PSNR:{(sum(p) / len(p)):.2f}, Avg SSIM:{(sum(s) / len(s)):.3f}")
