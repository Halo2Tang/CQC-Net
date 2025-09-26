import os
import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import hilbert as ht
from scipy.fftpack import ihilbert as iht
import pywt
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class QWTHelper:
    def __init__(self):
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bicubic')
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

    def reflect(self, x, minx, maxx):
        rng = maxx - minx
        rng_by_2 = 2 * rng
        mod = torch.fmod(x - minx, rng_by_2)
        normed_mod = torch.where(mod < 0, mod + rng_by_2, mod)
        return torch.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx

    def as_column_vector(self, v):
        v = torch.atleast_2d(v)
        return v.t() if v.shape[0] == 1 else v

    def as_row_vector(self, v):
        v = torch.atleast_2d(v)
        return v if v.shape[0] == 1 else v.t()

    def _centered(self, arr, newsize):
        currsize = torch.tensor(arr.shape, device=arr.device)
        startind = torch.div(currsize - newsize, 2, rounding_mode='floor')
        endind = startind + newsize
        slices = [slice(startind[k], endind[k]) for k in range(len(endind))]
        return arr[tuple(slices)]

    def _column_convolve(self, X, h):
        h = h.flatten()
        
        h_size = h.shape[0]
        out = torch.zeros_like(X)
        for idx in range(h_size):
            out += X * h[idx]
        out_shape = torch.tensor(X.shape, device=X.device)
        out_shape[2] = abs(X.shape[2] - h_size)
        return self._centered(out, out_shape)

    def _row_convolve(self, X, h):
        h = h.flatten()
        h_size = h.shape[0]
        out = torch.zeros_like(X)
        for idx in range(h_size):
            out += X * h[idx]
        out_shape = torch.tensor(X.shape, device=X.device)
        out_shape[3] = abs(X.shape[3] - h_size)
        return self._centered(out, out_shape)

    def colfilter(self, X, h):
        h = self.as_column_vector(h)
        batch, ch, r, _ = X.shape
        m = h.shape[0]
        m2 = m // 2
        xe = self.reflect(torch.arange(-m2, r + m2, device=X.device), -0.5, r - 0.5).long()
        return self._column_convolve(X[:, :, xe], h)

    def rowfilter(self, X, h):
        h = self.as_row_vector(h)
        batch, ch, _, c = X.shape
        m = h.shape[1]
        m2 = m // 2
        xe = self.reflect(torch.arange(-m2, c + m2, device=X.device), -0.5, c - 0.5).long()
        return self._row_convolve(X[:, :, :, xe], h)

class QWTForward(nn.Module):
    def __init__(self, device):
        super(QWTForward, self).__init__()
        self.device = device
        self.qwt_helper = QWTHelper()
        wavelet = pywt.Wavelet('coif5')
        gl = np.array(wavelet.dec_lo, dtype=np.float32)
        gh = np.array(wavelet.dec_hi, dtype=np.float32)
        fl = ht(gl)
        fh = ht(gh)
        self.gl = torch.from_numpy(gl).to(device)
        self.gh = torch.from_numpy(gh).to(device)
        self.fl = torch.from_numpy(fl).to(device)
        self.fh = torch.from_numpy(fh).to(device)

    def forward(self, image):
        helper = self.qwt_helper
        # Precompute column filters to reduce redundancy
        gl_col = helper.colfilter(image, self.gl)
        gh_col = helper.colfilter(image, self.gh)
        fl_col = helper.colfilter(image, self.fl)
        fh_col = helper.colfilter(image, self.fh)

        # Downsample once per column filter
        gl_col_ds = helper.downsample(gl_col)
        gh_col_ds = helper.downsample(gh_col)
        fl_col_ds = helper.downsample(fl_col)
        fh_col_ds = helper.downsample(fh_col)

        # Compute all combinations
        lglg = helper.rowfilter(gl_col_ds, self.gl)
        lghg = helper.rowfilter(gl_col_ds, self.gh)
        lglf = helper.rowfilter(gl_col_ds, self.fl)
        lghf = helper.rowfilter(gl_col_ds, self.fh)

        hglg = helper.rowfilter(gh_col_ds, self.gl)
        hghg = helper.rowfilter(gh_col_ds, self.gh)
        hglf = helper.rowfilter(gh_col_ds, self.fl)
        hghf = helper.rowfilter(gh_col_ds, self.fh)

        lflg = helper.rowfilter(fl_col_ds, self.gl)
        lfhg = helper.rowfilter(fl_col_ds, self.gh)
        lflf = helper.rowfilter(fl_col_ds, self.fl)
        lfhf = helper.rowfilter(fl_col_ds, self.fh)

        hflg = helper.rowfilter(fh_col_ds, self.gl)
        hfhg = helper.rowfilter(fh_col_ds, self.gh)
        hflf = helper.rowfilter(fh_col_ds, self.fl)
        hfhf = helper.rowfilter(fh_col_ds, self.fh)

        return (torch.cat((lglg, lflg, lglf, lflf), dim=1),
                torch.cat((lghg, lfhg, lghf, lfhf), dim=1),
                torch.cat((hglg, hflg, hglf, hflf), dim=1),
                torch.cat((hghg, hfhg, hghf, hfhf), dim=1))

class QWTInverse(nn.Module):
    def __init__(self, device):
        super(QWTInverse, self).__init__()
        self.device = device
        self.qwt_helper = QWTHelper()
        wavelet = pywt.Wavelet('coif5')
        gl = np.array(wavelet.dec_lo, dtype=np.float32)
        gh = np.array(wavelet.dec_hi, dtype=np.float32)
        fl = iht(gl)
        fh = iht(gh)
        self.gl = torch.from_numpy(gl).to(device)
        self.gh = torch.from_numpy(gh).to(device)
        self.fl = torch.from_numpy(fl).to(device)
        self.fh = torch.from_numpy(fh).to(device)

    def forward(self, LL, LH, HL, HH):
        helper = self.qwt_helper
        split_size = LL.size(1) // 4
        lglg, lflg, lglf, lflf = LL.split(split_size, dim=1)
        lghg, lfhg, lghf, lfhf = LH.split(split_size, dim=1)
        hglg, hflg, hglf, hflf = HL.split(split_size, dim=1)
        hghg, hfhg, hghf, hfhf = HH.split(split_size, dim=1)

        # First component
        t1 = helper.upsample(helper.rowfilter(lglg, self.gl))
        t2 = helper.upsample(helper.rowfilter(lghg, self.gh))
        t1 = helper.colfilter(t1 + t2, self.gl)
        t3 = helper.upsample(helper.rowfilter(hglg, self.gl))
        t4 = helper.upsample(helper.rowfilter(hghg, self.gh))
        t2 = helper.colfilter(t3 + t4, self.gh)
        first_component = t1 + t2

        # Second component
        t1 = helper.upsample(helper.rowfilter(lflg, self.gl))
        t2 = helper.upsample(helper.rowfilter(lfhg, self.gh))
        t1 = helper.colfilter(t1 + t2, self.fl)
        t3 = helper.upsample(helper.rowfilter(hflg, self.gl))
        t4 = helper.upsample(helper.rowfilter(hfhg, self.gh))
        t2 = helper.colfilter(t3 + t4, self.fh)
        second_component = t1 + t2

        # Third component
        t1 = helper.upsample(helper.rowfilter(lglf, self.fl))
        t2 = helper.upsample(helper.rowfilter(lghf, self.fh))
        t1 = helper.colfilter(t1 + t2, self.gl)
        t3 = helper.upsample(helper.rowfilter(hglf, self.fl))
        t4 = helper.upsample(helper.rowfilter(hghf, self.fh))
        t2 = helper.colfilter(t3 + t4, self.gh)
        third_component = t1 + t2

        # Fourth component
        t1 = helper.upsample(helper.rowfilter(lflf, self.fl))
        t2 = helper.upsample(helper.rowfilter(lfhf, self.fh))
        t1 = helper.colfilter(t1 + t2, self.fl)
        t3 = helper.upsample(helper.rowfilter(hflf, self.fl))
        t4 = helper.upsample(helper.rowfilter(hfhf, self.fh))
        t2 = helper.colfilter(t3 + t4, self.fh)
        fourth_component = t1 + t2

        y = first_component + second_component + third_component + fourth_component
        return (y - y.min()) / (y.max() - y.min())
    

# def compute_mse(img1, img2):
#     """计算两个图像的均方误差"""
#     return torch.mean((img1 - img2) ** 2).item()

# def compute_psnr(img1, img2, max_val=1.0):
#     """计算峰值信噪比"""
#     mse = compute_mse(img1, img2)
#     if mse == 0:
#         return float('inf')
#     return 10 * np.log10(max_val ** 2 / mse)

# def main():
#     # 加载输入图像
#     img_original = pywt.data.camera()
#     img = torch.from_numpy(img_original).unsqueeze(0).unsqueeze(0).float() / 255.0
#     img = img.to("cuda")
#     print("Input shape:", img.shape)

#     # 原代码（假设你有原代码的 QWTForward 和 QWTInverse）
#     # 这里需要替换为你的原代码实现
#     # qwt_orig = QWTForwardOrig("cuda")
#     # iqwt_orig = QWTInverseOrig("cuda")
#     # wavelets_orig = qwt_orig(img)
#     # LL_orig, LH_orig, HL_orig, HH_orig = wavelets_orig
#     # iqwt_out_orig = iqwt_orig(LL_orig, LH_orig, HL_orig, HH_orig)

#     # 修改后的代码
#     qwt = QWTForward("cuda")
#     iqwt = QWTInverse("cuda")
#     wavelets = qwt(img)
#     LL, LH, HL, HH = wavelets
#     print("LL shape:", LL.shape)
#     iqwt_out = iqwt(LL, LH, HL, HH)
#     print("Output shape:", iqwt_out.shape)

#     # 比较重建误差（假设原代码输出可用）
#     # iqwt_out_orig_cropped = iqwt_out_orig[:, :, :512, :512]
#     mse = compute_mse(img, iqwt_out)
#     psnr = compute_psnr(img, iqwt_out)
#     print(f"MSE between input and modified output: {mse:.6f}")
#     print(f"PSNR between input and modified output: {psnr:.2f} dB")

#     # 可视化
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
#     plt.title("Input Image")
#     plt.subplot(1, 3, 2)
#     plt.imshow(iqwt_out.squeeze().cpu().numpy(), cmap='gray')
#     plt.title("Reconstructed Image (Modified)")
#     plt.subplot(1, 3, 3)
#     plt.imshow((img - iqwt_out).abs().squeeze().cpu().numpy(), cmap='hot')
#     plt.title("Difference (Input - Reconstructed)")
#     plt.show()

# if __name__ == "__main__":
#     main()