# %%
"""
Fourier process combined with the techniques from the paper "Learning in Frequency Domain": https://arxiv.org/abs/2002.12416
"""

import time
import sys
import os
import logging

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, RandomHorizontalFlip
import kornia

# FFT utilities and data transforms
from utils import (
    fft2,
    ifft2,
    patchify,
    unpatchify,
    group_dct_components,
    ungroup_dct_components,
    ungroup_components_stat,
    get_dataset_stats,
)

sys.path.insert(0, os.getcwd())
from noise_schedulers import fourier_scheduler, fourier_exp_scheduler
from custom_datasets import DatasetFromDir
from fourier_process import FourierRunner, get_parser, parse_config, simple_ema
from pipelines.fddm_pipeline import FDDMPipeline
from models.ddpm_unet_LiF import ddpm_unet_LiF
from models.DiT_LiF import DiT_LiF_models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FDDMRunner(FourierRunner):
    def transform_to_scaled_fourier(self, image):
        assert image.shape[-3] in (1, 3), "Image must be grayscale or RGB"
        if image.shape[-3] == 3:
            image = kornia.color.rgb_to_ycbcr(image)
        image = image * 2 - 1
        return image

    def transform_from_scaled_fourier(self, scaled_image_batch):
        assert scaled_image_batch.shape[-3] in (1, 3), "Image must be grayscale or YCbCr"
        scaled_image_batch = (scaled_image_batch + 1) / 2
        if scaled_image_batch.shape[-3] == 3:
            scaled_image_batch = kornia.color.ycbcr_to_rgb(scaled_image_batch)
        return scaled_image_batch.clamp(0, 1)

    def get_trainset(self):
        transform_train = Compose([
            Resize(self.config.image_size),
            RandomHorizontalFlip(),
            CenterCrop(self.config.image_size),
            ToTensor(),
        ])

        if self.config.dataset == "CelebA":
            return DatasetFromDir(
                "/root/new/FDDM/data/img_align_celeba",
                self.config.image_size,
                transform=transform_train,
            )
        elif self.config.dataset == "CIFAR10":
            return datasets.CIFAR10(
                root="/root/new/FDDM/data", train=True, download=True, transform=transform_train
            )
        elif self.config.dataset == "FashionMNIST":
            return datasets.FashionMNIST(
                root="/root/new/FDDM/data", train=True, download=True, transform=transform_train
            )
        else:
            raise ValueError(
                "Dataset not supported. Please choose from CelebA, CIFAR10, FashionMNIST"
            )

    def get_model(self):
        img_size = self.config.image_size // self.config.patch_size
        if self.config.patch_size == self.config.image_size:
            freq_channels = self.config.channels
        else:
            freq_channels = self.config.channels * self.config.patch_size**2

        keep_dim = not (
            self.config.patch_size in (1, self.config.image_size)
        )

        if self.config.unet == "ddpm_unet_LiF":
            logging.info("Using ddpm_unet_LiF")
            model = ddpm_unet_LiF(
                resolution=self.config.image_size,
                T=self.config.num_train_timesteps,
                in_channels=freq_channels,
                ch=max(192, freq_channels),
                ch_mult=[1, 2, 2, 2],
                attn=[1],
                num_res_blocks=2,
                keep_dim=keep_dim,
                process_config=self.config,
                dropout=0.1,
            )
        elif self.config.unet == "DiT_LiF":
            model = DiT_LiF_models[f"{self.config.DiT_model}/{self.config.patch_size}"](
                input_size=self.config.image_size,
                in_channels=self.config.channels,
                num_classes=self.config.image_size,
                process_config=self.config,
            )
        else:
            raise ValueError("Unet type not recognized")

        logging.info(f"Number of parameters: {count_parameters(model)}")
        return model

    def get_pipeline(self, model, scheduler):
        return FDDMPipeline(
            process_config=self.config,
            unet=model,
            scheduler=scheduler,
            denoise_algo=self.config.denoise_algo,
            loss_type=self.config.loss_type,
            ddim_sigma=self.config.ddim_sigma,
        )

    @torch.no_grad()
    def get_noisy_images(self, clean_images, noise, timesteps):
        return self.scheduler.add_noise(clean_images, noise, timesteps)

    def sample(self, num, ema=False, seed=0, return_all_steps=False):
        pipeline = self.pipeline_ema if ema else self.pipeline
        return pipeline(
            batch_size=num,
            generator=torch.manual_seed(seed),
            num_inference_steps=self.config.num_inference_steps,
            pad=self.config.pad,
            return_image_and_fourier=True,
            transform_from_scaled_fourier=self.transform_from_scaled_fourier,
            return_all=return_all_steps,
        )

    def get_loss(
        self,
        timesteps,
        clean_images,
        reshaped_norm_clean_images_freq,
        model_out,
    ):
        if self.config.loss_type == "full_img":
            return F.mse_loss(clean_images, model_out)

        elif self.config.loss_type == "reshaped_freq":
            return F.mse_loss(
                reshaped_norm_clean_images_freq, model_out, reduction="none"
            )

        elif self.config.loss_type in (
            "reshaped_freq_full_img", "reshaped_freq_full_img_v2"
        ):
            patch_size = self.config.patch_size
            image_size = self.config.image_size
            out_patch = ungroup_dct_components(
                model_out,
                patch_size=patch_size,
                num_patches_per_dim=image_size // patch_size,
            )
            if self.config.loss_type == "reshaped_freq_full_img":
                out_patch = out_patch * self.reshaped_std + self.reshaped_mean
            out_image = unpatchify(ifft2(out_patch), image_size // patch_size)
            return F.mse_loss(clean_images, out_image, reduction="none")

        else:
            raise Exception(
                "loss_type must be full_img, patch_img, reshaped_freq, reshaped_freq_full_img"
            )

    def get_scheduler(self):
        if self.config.fermi_schedule == "linear":
            logging.info("Using linear fermi schedule")
            return fourier_scheduler.FourierScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                temperature=self.config.temperature,
                fermi_start=self.config.fermi_start,
                fermi_end=self.config.fermi_end,
                image_size=self.config.patch_size,
            )
        elif self.config.fermi_schedule == "exp":
            logging.info("Using exp fermi schedule")
            return fourier_exp_scheduler.FourierExpScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                temperature=self.config.temperature,
                fermi_start=self.config.fermi_start,
                fermi_end=self.config.fermi_end,
                image_size=self.config.patch_size,
                fermi_muliplier=self.config.fermi_muliplier,
            )
        else:
            raise ValueError("Unsupported fermi schedule type")

    def run(self):
        device = self.config.device
        patch_size = self.config.patch_size
        image_size = self.config.image_size
        input_channels = self.config.channels
        patch_num = (image_size // patch_size) ** 2

        mean, std = get_dataset_stats(
            dataset=self.config.dataset,
            patch_size=patch_size,
            dummy=not self.config.norm_freq_comps,
        )
        self.reshaped_mean = ungroup_components_stat(
            mean, input_channels, patch_size, image_size // patch_size
        ).to(device)
        self.reshaped_std = ungroup_components_stat(
            std, input_channels, patch_size, image_size // patch_size
        ).to(device)

        _step = self.init_step
        for epoch in range(self.config.epochs):
            for data in self.dataloader:
                clean_images = data[0].to(device)
                patch_clean = patchify(clean_images, patch_size)
                batch = patch_clean.shape[0]
                timesteps = self.get_train_timesteps(
                    0, self.get_num_train_timesteps(), batch, device
                )
                noise = self.get_noise(patch_clean.shape, patch_clean.device)

                # Replace DCT with FFT
                patch_clean_freq = fft2(patch_clean)
                norm_clean_freq = (patch_clean_freq - self.reshaped_mean) / self.reshaped_std

                noisy_freq = self.get_noisy_images(norm_clean_freq, noise, timesteps)

                reshaped_noisy = group_dct_components(
                    noisy_freq, patch_size, image_size // patch_size, input_channels
                )
                reshaped_clean = group_dct_components(
                    norm_clean_freq, patch_size, image_size // patch_size, input_channels
                )

                model_out = self.model(reshaped_noisy, timesteps)
                error = self.get_loss(timesteps, clean_images, reshaped_clean, model_out)
                loss = error.mean()

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    simple_ema(
                        self.accelerator.unwrap_model(self.model),
                        self.ema_model,
                        self.config.ema_decay,
                    )
                _step += 1
                if _step >= self.config.max_steps:
                    break
            if _step >= self.config.max_steps:
                break

        self.save_state("final")


if __name__ == "__main__":
    parser = get_parser()
    config = parse_config(parser)
    runner = FDDMRunner(config)
    runner.run()
