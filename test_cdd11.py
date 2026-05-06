import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset, DeblurTestDataset, LOLTestDataset, CDD11
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model_compos import R2R, R2RLocal
from utils.schedulers import *
import lightning.pytorch as pl
import torch.nn.functional as F

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

def test_CDD11(net, dataset, subset):
    output_path = testopt.output_path + subset
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name, degradation_type], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            temp = [0, 0, 0, 0]
            if "haze" in degradation_type[0]:
                temp[0] = 1
            if "rain" in degradation_type[0]:
                temp[1] = 1
            if "snow" in degradation_type[0]:
                temp[2] = 1
            if "low" in degradation_type[0]:
                temp[3] = 1

            interact_label = torch.tensor([temp], dtype=torch.float32).cuda()

            restored = patch_inference(degrad_patch, net, tile=None,  interact_label=interact_label)
            restored = restored[:, :, :clean_patch.shape[-2], :clean_patch.shape[-1]]

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path  + '/'+clean_name[0].split("/")[-1])

        return psnr.avg, ssim.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=1,
                        help='1 for single, 2 for double, 3 for triple, 4 for all')
    parser.add_argument('--output_path', type=str, default="output/CDD11/", help='output save path')
    parser.add_argument('--data_file_dir', type=str, default="data/Test/CDD11/", help='save path of test noisy images')
    parser.add_argument('--ckpt_name', type=str, default="train_ckpt_compos_finetune/", help='checkpoint save path')
    parser.add_argument('--prompt_dir', type=str, default="save_prompts_compos/", help='prompt save path')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_name

    print("CKPT name : {}".format(ckpt_path))

    for epoch in range(29, 30): # best: 202 205
        name = 'last'
        net = R2RLocal(ckpt_path=ckpt_path, prompts_path=testopt.prompt_dir, prompts_name=name, train_mode="finetune").cuda()

        print(str(epoch)+"---------------------------------------------------------------------")

        net.eval()

        test_single = {"single":["low", "haze", "rain", "snow"]}
        test_double = {"double":["low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow"]}
        test_triple = {"triple":["low_haze_rain", "low_haze_snow"]}
        deg_type = {1:test_single,2:test_double,3:test_triple}

        p=[]
        s=[]

        if testopt.mode==4:
            for i in range(1, 4):
                key = list(deg_type[i].keys())[0]

                print("--------> Testing on", key)
                for subset in deg_type[i][key]:
                    dataset = CDD11(testopt, split="test", subset=subset)

                    psnr, ssim = test_CDD11(net, dataset, subset=subset)
                    print("{}--------> PSNR={:.2f} SSIM={:.3f}".format(subset, psnr, ssim))
                    p.append(psnr)
                    s.append(ssim)
            print(f"Avg PSNR:{(sum(p) / len(p)):.2f}, Avg SSIM:{(sum(s) / len(s)):.3f}")
        else:

            key = list(deg_type[testopt.mode].keys())[0]

            print("--------> Testing on", key)
            for subset in deg_type[testopt.mode][key]:

                dataset = CDD11(testopt, split="test", subset=subset)

                psnr, ssim = test_CDD11(net, dataset, subset=subset)
                print("{}--------> PSNR={:.2f} SSIM={:.3f}".format(subset, psnr, ssim))
                p.append(psnr)
                s.append(ssim)
            print(f"Avg PSNR:{(sum(p) / len(p)):.2f}, Avg SSIM:{(sum(s) / len(s)):.3f}")
