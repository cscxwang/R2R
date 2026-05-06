
import time
import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

import numpy as np
import cv2

# def y_channel_psnr(img1, img2):
    # # 将RGB图像转换为YCbCr格式
    # ycbcr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    # ycbcr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
    # y1 = ycbcr1[:,:,0]
    # y2 = ycbcr2[:,:,0]
    #
    # # 计算:ml-search-more[均方误差]{text="均方误差"}（MSE）
    # mse = np.mean((y1 - y2) ** 2)
    #
    # # 计算PSNR值
    # max_val = 1  # 8位图像最大值
    # psnr = 20 * np.log10(max_val ** 2 / mse)
    # return psnr




# y_channel_psnr
# def compute_psnr_ssim(recoverd, clean):
#     assert recoverd.shape == clean.shape
#     recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
#     clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
#
#     recoverd = recoverd.transpose(0, 2, 3, 1)
#     clean = clean.transpose(0, 2, 3, 1)
#
#     psnr = 0
#     ssim = 0
#
#     for i in range(recoverd.shape[0]):
#
#
#         img1_y = cv2.cvtColor(recoverd[i], cv2.COLOR_RGB2YCrCb)[:, :, 0]
#         img2_y = cv2.cvtColor(clean[i], cv2.COLOR_RGB2YCrCb)[:, :, 0]
#
#         psnr += peak_signal_noise_ratio(img1_y, img2_y, data_range=1)
#         ssim += structural_similarity(img1_y, img2_y, data_range=1)
#
#
#
#     return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        # ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=-1)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0