import logging
from pathlib import Path
import random

import numpy as np
import skimage.metrics
import torch


def get_logger(verbose: bool, save_dir: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(message)s',
        datefmt='%Y%m%dT%H%M%SZ'
    )
    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(save_dir / 'log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def img_metrics(img_true, img_test):
    if np.all(img_true == img_test):
        return {
            'SSIM': 1.,
            'PSNR': np.inf,
            'NRMSE': 0.
        }
    else:
        data_range = np.max(img_true)
        return {
            'SSIM': skimage.metrics.structural_similarity(img_true, img_test, data_range=data_range),
            'PSNR': skimage.metrics.peak_signal_noise_ratio(img_true, img_test, data_range=data_range),
            'NRMSE': skimage.metrics.normalized_root_mse(img_true, img_test)
        }
