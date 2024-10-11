import argparse
import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.utils import make_grid
from tqdm import tqdm

from .models.aft import PAFT_DC
from .models.cunet import CUNet
from .datasets import AFTNetSliceDataset, AFTNetVolumeDataset
import torchmri
from . import utils


class RACUNet1(Module):
    def __init__(self) -> None:
        super().__init__()
        self.acunet = CUNet(
            in_channels=4,
            out_channels=4,
            layer_channels=[32, 64, 128, 256, 512],
            attention=True
        )

    def forward(self, isp_in: Tensor, ksp_in: Tensor) -> Tensor:
        isp_pred = isp_in + self.acunet(isp_in)
        ksp_pred = torchmri.fft.fftn(isp_pred, dim=(-2, -1))
        ksp_pred = apply_k_space_consistency(ksp_in, ksp_pred)
        isp_pred = torchmri.fft.ifftn(ksp_pred, dim=(-2, -1))
        return isp_pred

class RACUNet2(Module):
    def __init__(self) -> None:
        super().__init__()
        self.acunet = CUNet(
            in_channels=4,
            out_channels=4,
            layer_channels=[32, 64, 128, 256, 512],
            attention=True
        )

    def forward(self, ksp_in: Tensor) -> Tensor:
        ksp_out = ksp_in + self.acunet(ksp_in)
        ksp_out = apply_k_space_consistency(ksp_in, ksp_out)
        return ksp_out

def apply_k_space_consistency(k_space_under_gt, k_space_output):
    """
    Apply k-space data consistency by replacing the values in the k-space output
    with the undersampled ground truth values where the ground truth is not zero.

    Args:
        k_space_gt (torch.Tensor): The undersampled ground truth k-space data.
        k_space_output (torch.Tensor): The output k-space data from the model.

    Returns:
        torch.Tensor: The k-space output with data consistency applied.
    """
    # Ensure the tensors are on the same device
    k_space_under_gt = k_space_under_gt.to(k_space_output.device)

    # Create a mask where the undersampled ground truth is not zero
    mask = k_space_under_gt != 0

    # Apply the mask to update the k-space output with the ground truth values
    k_space_output[mask] = k_space_under_gt[mask]

    return k_space_output


class RACUNet_PAFT_RACUNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.aft = PAFT_DC(320)
        self.racunet1 = RACUNet1()
        self.racunet2 = RACUNet2()

    def forward(self, ksp_in: Tensor) -> Tensor:
        return self.racunet1(self.aft(self.racunet2(ksp_in)), ksp_in)


class Pipeline:
    def __init__(self, save_dir: Path, debug: bool = False, verbose: bool = True) -> None:
        self.logger = utils.get_logger(verbose, save_dir)
        self.writer = SummaryWriter(save_dir)

        self.logger.info(f'seed: 42')
        utils.setup_seed(42)

        self.logger.info(f'PID = {os.getpid()}')

        self.train_loader, self.val_loader, self.test_loader = self._get_loader()

        self.model = RACUNet_PAFT_RACUNet()
        self.model.aft.load_state_dict(
            torch.load('/home/yanting/projects/AFT-Netv6/240712_PAFT_acc/runs/best_model.pt', map_location='cpu')
        )
        self.model.racunet1.load_state_dict(
            torch.load('/home/yanting/projects/AFT-Netv6/240708_RACUNet_i_acc_data/runs/best_model.pt', map_location='cpu')
        )
        self.model.racunet2.load_state_dict(
            torch.load('/home/yanting/projects/AFT-Netv6/240704_RACUNet_k_acc_data-consist/runs/best_model.pt', map_location='cpu')
        )

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=1, eta_min=1e-5)

        self.debug = debug
        if self.debug:
            self.logger.info('debug mode')
        self.verbose = verbose
        self.save_dir = save_dir

    @staticmethod
    def _get_loader():
        train_dataset = AFTNetSliceDataset(
            'data/v0', 'training', ['1x', '2x', '4x', '4x', '8x', '8x'], [0.], (640, 320))
        val_dataset = AFTNetVolumeDataset(
            'data/v0', 'validation', ['4x'], [0.], (640, 320))
        test_dataset = AFTNetVolumeDataset(
            'data/v0', 'test', ['4x'], [0.], (640, 320))

        train_loader = DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=None, shuffle=False, num_workers=1)
        test_loader = DataLoader(
            test_dataset, batch_size=None, shuffle=False, num_workers=1)
        return train_loader, val_loader, test_loader

    @staticmethod
    def _get_imgs(img_in: Tensor, img_pred: Tensor, img_true: Tensor, metric: dict[str, float]):
        def _add_text(im: Tensor, text: str):
            im = v2.functional.to_pil_image(im)
            draw = ImageDraw.Draw(im)

            xy = (0, 0)
            fill = 'white'
            font = ImageFont.truetype('FreeMono.ttf', size=20)
            bbox = draw.textbbox(xy, text, font)
            draw.rectangle(bbox, fill='black')
            draw.text(xy, text, fill, font)
            return v2.functional.pil_to_tensor(im)
        vmax = img_true.max()
        img_in = img_in.div(vmax).clip(0, 1).mul(
            255).repeat(3, 1, 1).to(torch.uint8)
        img_pred = img_pred.div(vmax).clip(0, 1).mul(
            255).repeat(3, 1, 1).to(torch.uint8)
        img_true = img_true.div(vmax).mul(255).repeat(3, 1, 1).to(torch.uint8)

        s = []
        for k, v in metric.items():
            s.append(f'{k} = {v:.3f}')

        img_in = _add_text(img_in, 'in')
        img_pred = _add_text(img_pred, 'pred\n' + '\n'.join(s))
        img_true = _add_text(img_true, 'true')
        return img_in, img_pred, img_true

    def __call__(self):
        for self.epoch in range(50):
            self.logger.info(f'Epoch {self.epoch}')
            self.train_epoch()
            self.val_epoch()
            if self.debug and self.epoch == 1:
                break
        self.test_epoch()
        self.writer.close()

    def train_epoch(self):
        self.model.train()

        iter = len(self.train_loader)
        losses = {
            'mse_loss': []
        }

        for i, batch in enumerate(tqdm(self.train_loader, desc='Train', disable=not self.verbose, dynamic_ncols=True)):
            if self.debug and i == 2:
                break

            batch: dict[str, Tensor]
            isp_in, isp_true, ksp_in, ksp_true = batch['ispace_under'], batch[
                'ispace'], batch['kspace_under'], batch['kspace']

            isp_pred = self.model(ksp_in.to(self.device))
            img_pred = torchmri.utils.rss(isp_pred, dim=-3)
            img_true = torchmri.utils.rss(isp_true.to(self.device), dim=-3)

            mse_loss = F.mse_loss(
                torch.view_as_real(isp_pred),
                torch.view_as_real(isp_true).to(self.device),
                reduction='sum'
            ) + F.mse_loss(
                img_pred, img_true, reduction='sum'
            )
            loss = mse_loss
            for k in losses.keys():
                losses[k].append(locals()[k].item())
                self.writer.add_scalar(
                    f'{k}/train', losses[k][-1], self.epoch * iter + i)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar(
                'lr/train', self.scheduler.get_last_lr()[0], self.epoch * iter + i)
            self.scheduler.step(self.epoch + i / iter)

        s = []
        for k, v in losses.items():
            s.append(f'{k} = {np.mean(v):.3E} ± {np.std(v):.3E}')
        self.logger.info('Train: ' + ', '.join(s))
        self.logger.info(f'Train: lr={self.scheduler.get_last_lr()[0]:.2E}')

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()

        metrics = []

        for i, batch in enumerate(tqdm(self.val_loader, desc='Val', disable=not self.verbose, dynamic_ncols=True)):
            if self.debug and i == 2:
                break

            batch: dict[str, Tensor]
            isp_in, isp_true, ksp_in, ksp_true = batch['ispace_under'], batch[
                'ispace'], batch['kspace_under'], batch['kspace']

            isp_pred = self.model(ksp_in.to(self.device))

            img_in = torchmri.utils.rss(isp_in[:, :, 160:-160], dim=-3)
            img_pred = torchmri.utils.rss(
                isp_pred[:, :, 160:-160], dim=-3).cpu()
            img_true = torchmri.utils.rss(isp_true[:, :, 160:-160], dim=-3)

            grids = []
            for j in range(img_true.shape[0]):
                metrics.append(utils.img_metrics(
                    img_true[j].numpy(), img_pred[j].numpy()))
                if i == 0:
                    _img_in, _img_pred, _img_true = self._get_imgs(
                        img_in[j], img_pred[j], img_true[j], metrics[-1])
                    grids.extend([_img_in, _img_pred, _img_true])
            if i == 0:
                grids = torch.stack(grids, dim=0)
                grids = make_grid(grids, nrow=3)
                self.writer.add_image(f'image/val', grids, self.epoch)

        metrics = pd.DataFrame.from_records(metrics)
        s = []
        for k in metrics.columns:
            v = metrics[k].to_list()
            mean, std = np.mean(v), np.std(v)
            s.append(f'{k} = {mean:.3f} ± {std:.3f}')
            self.writer.add_scalar(f'{k}/val', mean, self.epoch)
        self.logger.info('Val: ' + ', '.join(s))

        psnr = np.mean(metrics['PSNR'].to_list())
        if not hasattr(self, 'best_metric') or psnr > self.best_metric:
            self.best_metric = psnr
            torch.save(self.model.state_dict(),
                       self.save_dir / 'best_model.pt')

    @torch.no_grad()
    def test_epoch(self):
        self.model.load_state_dict(torch.load(
            self.save_dir / 'best_model.pt', map_location=self.device))
        self.model.eval()

        metrics = []

        for i, batch in enumerate(tqdm(self.test_loader, desc='Test', disable=not self.verbose, dynamic_ncols=True)):
            if self.debug and i == 2:
                break

            batch: dict[str, Tensor]
            isp_in, isp_true, ksp_in, ksp_true = batch['ispace_under'], batch[
                'ispace'], batch['kspace_under'], batch['kspace']

            isp_pred = self.model(ksp_in.to(self.device))

            img_true = torchmri.utils.rss(
                isp_true[:, :, 160:-160], dim=-3).numpy()
            img_pred = torchmri.utils.rss(
                isp_pred[:, :, 160:-160], dim=-3).cpu().numpy()
            for j in range(img_true.shape[0]):
                metrics.append(utils.img_metrics(img_true[j], img_pred[j]))

        metrics = pd.DataFrame.from_records(metrics)
        s = []
        for k in metrics.columns:
            v = metrics[k].to_list()
            mean, std = np.mean(v), np.std(v)
            s.append(f'{k} = {mean:.3f} ± {std:.3f}')
        self.logger.info('Test: ' + ', '.join(s))

        metrics.to_csv(self.save_dir / 'metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    save_dir = Path('runs')
    save_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(save_dir=save_dir, debug=args.debug)
    pipeline()


if __name__ == '__main__':
    main()
