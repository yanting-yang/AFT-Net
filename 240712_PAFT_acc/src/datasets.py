from pathlib import Path
import re

from fastmri.data import subsample
import h5py
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
import torch

MASKARGS = {
    'all': {'center_fractions': [0.16, 0.08, 0.04, 0.02], 'accelerations': [2, 4, 8, 16]},
    '2x': {'center_fractions': [0.16], 'accelerations': [2]},
    '4x': {'center_fractions': [0.08], 'accelerations': [4]},
    '8x': {'center_fractions': [0.04], 'accelerations': [8]},
    '16x': {'center_fractions': [0.02], 'accelerations': [16]}
}


class H5Dataset:
    def __init__(self, root_folder: str, section: str) -> None:
        self.pathes = sorted((Path(root_folder) / section).glob('*.h5'))


class AFTNetSliceDataset(H5Dataset):
    def __init__(
        self,
        root_folder: str,
        section: str,
        acc_rates: list[str],
        noise_scales: list[float],
        newshape: tuple[int, int]
    ) -> None:
        super().__init__(root_folder, section)
        self.acc_rates = acc_rates
        self.noise_scales = noise_scales
        self.newshape = newshape

        self.path_slice_idx = list()
        for path_idx, p in enumerate(self.pathes):
            num_slice = int(
                re.search('([0-9]+),([0-9]+),([0-9]+),([0-9]+)', p.name).group(1))
            for slice_idx in range(num_slice - 5):
                self.path_slice_idx.append([path_idx, slice_idx])

    def __len__(self):
        return len(self.path_slice_idx)

    def __getitem__(self, idx):
        path_idx, slice_idx = self.path_slice_idx[idx]
        path = self.pathes[path_idx]
        with h5py.File(path) as f:
            kspace = f['kspace'][slice_idx, :, :, :]  # (C,H,W)
            max_value = f.attrs['max']
            kspace /= max_value
        ispace = ifft2(kspace, norm='ortho').astype(np.complex64)

        if (ispace.shape[-2], ispace.shape[-1]) != self.newshape:
            ispace = reshape2D(ispace, self.newshape)
            kspace = fft2(ispace, norm='ortho').astype(np.complex64)

        acc_rate = np.random.choice(self.acc_rates)
        if acc_rate == '1x':
            mask = np.ones(self.newshape[-1])[np.newaxis, np.newaxis, :]
            mask = mask.astype(np.float32)
        else:
            maskfunc = subsample.EquispacedMaskFractionFunc(**MASKARGS[acc_rate])
            center_mask, accel_mask, num_low_frequencies = maskfunc.sample_mask(
                shape=(self.newshape[-1], 1), offset=1
            )
            mask = torch.max(
                center_mask, accel_mask
            ).squeeze().numpy()[np.newaxis, np.newaxis, :].astype(np.float32)

        noise_scale = np.random.choice(self.noise_scales)
        if noise_scale == 0:
            noise = 0
        else:
            noise = np.random.normal(0, noise_scale, kspace.shape) \
                + 1j * np.random.normal(0, noise_scale, kspace.shape)
            noise = noise.astype(np.complex64)

        if acc_rate == '1x' and noise_scale == 0:
            kspace_trans = kspace.copy()
            ispace_trans = ispace.copy()
        else:
            kspace_trans = (kspace + noise) * mask
            ispace_trans = ifft2(kspace_trans, norm='ortho').astype(np.complex64)
        return {
            'kspace': kspace,
            'ispace': ispace,
            'kspace_under': kspace_trans,
            'ispace_under': ispace_trans,
            'mask': mask
        }


class AFTNetVolumeDataset(H5Dataset):
    def __init__(
        self,
        root_folder: str,
        section: str,
        acc_rates: list[str],
        noise_scales: list[float],
        newshape: tuple[int, int]
    ) -> None:
        super().__init__(root_folder, section)
        self.acc_rates = acc_rates
        self.noise_scales = noise_scales
        self.newshape = newshape

    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, idx):
        path_idx = idx
        path = self.pathes[path_idx]
        with h5py.File(path) as f:
            kspace = f['kspace'][:-5, :, :, :]  # (S,C,H,W)
            max_value = f.attrs['max']
            kspace /= max_value
        ispace = ifft2(kspace, norm='ortho').astype(np.complex64)

        if (ispace.shape[-2], ispace.shape[-1]) != self.newshape:
            ispace = reshape2D(ispace, self.newshape)
            kspace = fft2(ispace, norm='ortho').astype(np.complex64)

        acc_rate = np.random.choice(self.acc_rates)
        if acc_rate == '1x':
            mask = np.ones((kspace.shape[0], 1, 1, self.newshape[-1]))
            mask = mask.astype(np.float32)
        else:
            maskfunc = subsample.EquispacedMaskFractionFunc(**MASKARGS[acc_rate])
            center_mask, accel_mask, num_low_frequencies = maskfunc.sample_mask(
                shape=(self.newshape[-1], 1), offset=1
            )
            mask = torch.max(
                center_mask, accel_mask
            ).squeeze().numpy()[np.newaxis, np.newaxis, np.newaxis, :] \
                * np.ones((kspace.shape[0], 1, 1, self.newshape[-1]))
            mask = mask.astype(np.float32)

        noise_scale = np.random.choice(self.noise_scales)
        if noise_scale == 0:
            noise = 0
        else:
            np.random.seed(np.iinfo(np.uint32).max - 1 - idx)
            noise = np.random.normal(0, noise_scale, kspace.shape)\
                + 1j * np.random.normal(0, noise_scale, kspace.shape)
            noise = noise.astype(np.complex64)

        if acc_rate == '1x' and noise_scale == 0:
            kspace_trans = kspace.copy()
            ispace_trans = ispace.copy()
        else:
            kspace_trans = (kspace + noise) * mask
            ispace_trans = ifft2(kspace_trans, norm='ortho').astype(np.complex64)
        return {
            'kspace': kspace,
            'ispace': ispace,
            'kspace_under': kspace_trans,
            'ispace_under': ispace_trans,
            'mask': mask
        }


class AUTOMAPSliceDataset(H5Dataset):
    def __init__(
        self,
        root_folder: str,
        section: str,
        features_in: int
    ) -> None:
        super().__init__(root_folder, section)
        self.features_in = features_in

        self.path_slice_idx = list()
        for path_idx, p in enumerate(self.pathes):
            num_slice = int(re.search('\(([0-9]+),([0-9]+),([0-9]+),([0-9]+)\)', p.name).group(1))
            match = re.search('\(([0-9]+\.[0-9]+),([0-9]+\.[0-9]+),([0-9]+\.[0-9]+)\)', p.name)
            h_res, w_res = float(match.group(1)), float(match.group(2))
            for slice_idx in range(num_slice - 5):
                self.path_slice_idx.append([path_idx, slice_idx, h_res, w_res])

    def __len__(self):
        return len(self.path_slice_idx)

    def __getitem__(self, idx):
        path_idx, slice_idx, h_res, w_res = self.path_slice_idx[idx]
        path = self.pathes[path_idx]
        with h5py.File(path) as f:
            kspace = f['kspace'][slice_idx, :, :, :]    # (C,H,W)
            img = f['reconstruction_rss'][slice_idx]    # (H,W)
            max_value = f.attrs['max']
        kspace /= max_value
        img /= max_value

        C, H, W = kspace.shape
        ispace = ifft2(kspace)
        ispace = ndimage.zoom(ispace, (1, h_res * self.features_in / 192, w_res * self.features_in / 192))
        ispace = reshape2D(ispace, (self.features_in, self.features_in))
        kspace = fft2(ispace).astype(np.complex64)
        kspace = np.reshape(kspace, (C, -1))
        kspace = np.transpose(kspace, axes=(1, 0))
        U, _, _ = np.linalg.svd(kspace, full_matrices=False)
        kspace = U[:, 0]
        kspace = np.reshape(kspace, (1, self.features_in, self.features_in))

        img = ndimage.zoom(img, (h_res * self.features_in / 192, w_res * self.features_in / 192))
        img = reshape2D(img, (self.features_in, self.features_in))
        img = img[np.newaxis]
        return kspace, img


class AUTOMAPVolumeDataset(H5Dataset):
    def __init__(
        self,
        root_folder: str,
        section: str,
        features_in: int
    ) -> None:
        super().__init__(root_folder, section)
        self.features_in = features_in

        self.path_slice_idx = list()
        for path_idx, p in enumerate(self.pathes):
            match = re.search('([0-9]+\.[0-9]+),([0-9]+\.[0-9]+),([0-9]+\.[0-9]+)', p.name)
            h_res, w_res = float(match.group(1)), float(match.group(2))
            self.path_slice_idx.append([path_idx, h_res, w_res])

    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, idx):
        path_idx, h_res, w_res = self.path_slice_idx[idx]
        path = self.pathes[path_idx]

        with h5py.File(path) as f:
            kspace = f['kspace'][:, :, :, :]    # (S,C,H,W)
            img = f['reconstruction_rss'][:]    # (S,H,W)
            max_value = f.attrs['max']
        kspace /= max_value
        img /= max_value

        S, C, H, W = kspace.shape
        ispace = ifft2(kspace)
        ispace = ndimage.zoom(ispace, (1, 1, h_res * self.features_in / 192, w_res * self.features_in / 192))
        ispace = reshape2D(ispace, (self.features_in, self.features_in))
        kspace = fft2(ispace).astype(np.complex64)
        kspace = np.reshape(kspace, (S, C, -1))
        kspace = np.transpose(kspace, axes=(0, 2, 1))
        U, _, _ = np.linalg.svd(kspace, full_matrices=False)
        kspace = U[..., 0]
        kspace = np.reshape(kspace, (S, 1, self.features_in, self.features_in))

        img = ndimage.zoom(img, (1, h_res * self.features_in / 192, w_res * self.features_in / 192))
        img = reshape2D(img, (self.features_in, self.features_in))
        img = img[:, np.newaxis]
        return kspace, img


def fft2(a, axes=(-2, -1), norm='ortho'):
    a = np.fft.ifftshift(a, axes=axes)
    a = np.fft.fft2(a, axes=axes, norm=norm)
    a = np.fft.fftshift(a, axes=axes)
    return a


def ifft2(a, axes=(-2, -1), norm='ortho'):
    a = np.fft.ifftshift(a, axes=axes)
    a = np.fft.ifft2(a, axes=axes, norm=norm)
    a = np.fft.fftshift(a, axes=axes)
    return a


def reshape2D(a: NDArray, newshape: tuple[int, int]):
    new_a = np.zeros(a.shape[:-2] + newshape, dtype=a.dtype)
    slc_in = list()
    slc_out = list()
    for shape_in, shape_out in zip(a.shape, new_a.shape):
        if shape_in > shape_out:
            slc_out.append(slice(None))
            slc_in.append(slice(
                (shape_in - shape_out) // 2,
                (shape_in - shape_out) // 2 + shape_out
            ))
        elif shape_in < shape_out:
            slc_in.append(slice(None))
            slc_out.append(slice(
                (shape_out - shape_in) // 2,
                (shape_out - shape_in) // 2 + shape_in
            ))
        else:
            slc_in.append(slice(None))
            slc_out.append(slice(None))
    new_a[(*slc_out, )] = a[(*slc_in, )]
    return new_a
