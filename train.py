"""
3D CycleGAN
"""

import sys
from datetime import datetime, timedelta
from itertools import chain
from os import makedirs
from os.path import join
from shutil import copy
from time import perf_counter

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import config
from batch_dataset import BatchDataset
from loss_functions import ResidualLoss
from model import Discriminator, Generator
from utils import (
    DecayLR,
    GenericLogger,
    ReplayBuffer,
    plot_disc_gen_loss,
    plot_loss,
    print_with_timestamp,
)


def copy_sitk_metadata(image: sitk.Image, ref: sitk.Image) -> sitk.Image:
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


def copy_config(base, target):
    copy(base, target)


def make_output_directories(experiment_dir):
    makedirs(experiment_dir, exist_ok=True)
    makedirs(join(experiment_dir, "A"), exist_ok=True)
    makedirs(join(experiment_dir, "B"), exist_ok=True)
    makedirs(join(experiment_dir, "weights"), exist_ok=True)


def inverse_rescaling(img, mina, maxa):
    """Inverse transformation from [-1, +1] to [min_orig_val, max_orig_val].

    Args:
        img (numpy.ndarray): Input image array
        mina (float): Minimum original value
        maxa (float): Maximum original value

    Returns:
        numpy.ndarray: Original image (inverse of scaled image)
    """

    minv = img.min()
    maxv = img.max()

    return (img - minv) * (maxa - mina) / (maxv - minv) + mina


def inverse_transform(image: np.array, meta_image: sitk.Image) -> sitk.Image:
    # Get image array
    meta_image_arr = sitk.GetArrayFromImage(meta_image)
    # Inverse rescaling
    min_intensity = np.min(meta_image_arr)
    max_intensity = np.max(meta_image_arr)
    image = inverse_rescaling(image, min_intensity, max_intensity)
    # Create sitk object
    image = np.transpose(image)
    image = sitk.GetImageFromArray(image)
    # Set meta data
    image = copy_sitk_metadata(image, meta_image)
    return image


def load_checkpoint(checkpoint_file, model, optimizer, scheduler):
    print("=> Loading checkpoint...", end=" ")
    checkpoint_start = perf_counter()
    model.load_state_dict(torch.load(checkpoint_file)["model_state_dict"])
    optimizer.load_state_dict(torch.load(checkpoint_file)["optimizer_state_dict"])
    scheduler.load_state_dict(torch.load(checkpoint_file)["scheduler_state_dict"])
    checkpoint_end = perf_counter()
    checkpoint_time = checkpoint_end - checkpoint_start
    checkpoint_time = timedelta(seconds=checkpoint_time)
    print(f"Done. Loading took {checkpoint_time.seconds} seconds")


def load_csv_checkpoints(checkpoint_file):
    print(f"=> Loading csv checkpoint: {checkpoint_file}")
    checkpoints = pd.read_csv(checkpoint_file)
    checkpoints = checkpoints.iloc[:, 1].tolist()
    return checkpoints


def save_checkpoint(state, filename):
    print("=> Saving checkpoint...", end=" ")
    checkpoint_start = perf_counter()
    torch.save(state, filename)
    checkpoint_end = perf_counter()
    checkpoint_time = checkpoint_end - checkpoint_start
    checkpoint_time = timedelta(seconds=checkpoint_time)
    print(f"Done. Saving took {checkpoint_time.seconds} seconds")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train():
    """Train function"""

    # Start time
    start_time = perf_counter()

    # Set deterministic training for reproducibility
    set_determinism(seed=config.SEED)

    # Enable builtin hardware optimization
    cudnn.benchmark = config.CUDNNBENCHMARK

    # Make output directories
    experiment_dir = join(config.OUTPUT_DIR, config.EXPERIMENT_ID)
    make_output_directories(experiment_dir)

    # Initialize logger
    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
    f = open(join(experiment_dir, f"training_log_{timestamp}.txt"), "w")
    sys.stdout = GenericLogger(sys.stdout, f)

    # Copy run config file to output
    copy_config("config.py", experiment_dir)

    # Transforms
    transforms = config.transforms

    # Train dataset
    dataset = BatchDataset(
        root=config.DATA_DIR, transform=transforms, unpaired=True, mode="train"
    )

    # Train loader
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )

    # Uncomment this to check proper data loading
    # from monai.utils import first
    # first(dataloader)
    # im = first(dataloader)

    # Print dataset characteristics
    dataset.print_keys()

    # Device
    device = config.DEVICE

    # Create model
    netD_A = Discriminator(in_channels=1).to(device)
    netD_B = Discriminator(in_channels=1).to(device)
    netG_A2B = Generator(in_channels=1, num_residuals=9).to(device)
    netG_B2A = Generator(in_channels=1, num_residuals=9).to(device)

    # Use all available GPUs
    netG_A2B = nn.DataParallel(netG_A2B).to(device)
    netG_B2A = nn.DataParallel(netG_B2A).to(device)
    netD_A = nn.DataParallel(netD_A).to(device)
    netD_B = nn.DataParallel(netD_B).to(device)

    # Define loss function and optimizer
    cycle_loss = ResidualLoss().to(device)
    identity_loss = ResidualLoss().to(device)
    adversarial_loss = nn.MSELoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(
        chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    optimizer_D_A = optim.Adam(
        netD_A.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    optimizer_D_B = optim.Adam(
        netD_B.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )

    # Learning rate scheme
    lambda_lr = DecayLR(
        epochs=config.EPOCHS,
        offset=config.DECAY_OFFSET,
        decay_epochs=config.DECAY_EPOCHS,
    ).step
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lambda_lr)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lambda_lr)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lambda_lr)

    # Automatic mixed precision
    scaler = GradScaler()

    # Buffer
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    Loss_D = []
    Loss_G = []
    Loss_G_identity = []
    Loss_G_GAN = []
    Loss_G_cycle = []

    start_epoch = 1

    # Continue training / load model
    if config.LOAD_MODEL is True:
        start_epoch = config.CONTINUE_FROM_EPOCH
        if start_epoch == config.EPOCHS:
            print("TRAINING ALREADY COMPLETED")
            sys.exit()
        print("CONTINUING TRAINING")
        print(f"LOADING CHECKPOINT FROM EPOCH: {config.CONTINUE_FROM_EPOCH}")
        load_checkpoint(
            join(
                experiment_dir,
                "weights",
                f"netG_A2B_epoch_{config.CONTINUE_FROM_EPOCH}.pth.tar",
            ),
            netG_A2B,
            optimizer_G,
            lr_scheduler_G,
        )
        load_checkpoint(
            join(
                experiment_dir,
                "weights",
                f"netG_B2A_epoch_{config.CONTINUE_FROM_EPOCH}.pth.tar",
            ),
            netG_B2A,
            optimizer_G,
            lr_scheduler_G,
        )
        load_checkpoint(
            join(
                experiment_dir,
                "weights",
                f"netD_A_epoch_{config.CONTINUE_FROM_EPOCH}.pth.tar",
            ),
            netD_A,
            optimizer_D_A,
            lr_scheduler_D_A,
        )
        load_checkpoint(
            join(
                experiment_dir,
                "weights",
                f"netD_B_epoch_{config.CONTINUE_FROM_EPOCH}.pth.tar",
            ),
            netD_B,
            optimizer_D_B,
            lr_scheduler_D_B,
        )

        Loss_D = load_csv_checkpoints(join(experiment_dir, "Loss_D.csv"))
        Loss_G = load_csv_checkpoints(join(experiment_dir, "Loss_G.csv"))
        Loss_G_identity = load_csv_checkpoints(
            join(experiment_dir, "Loss_G_identity.csv")
        )
        Loss_G_GAN = load_csv_checkpoints(join(experiment_dir, "Loss_G_GAN.csv"))
        Loss_G_cycle = load_csv_checkpoints(join(experiment_dir, "Loss_G_cycle.csv"))
    else:
        print_with_timestamp("START TRAINING\n")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        epoch_start = perf_counter()

        print_with_timestamp(f"Epoch {epoch}/{config.EPOCHS}")
        print_with_timestamp(f"Learning rate Disc A: {get_lr(optimizer_D_A):.6f}")
        print_with_timestamp(f"Learning rate Disc B: {get_lr(optimizer_D_B):.6f}")
        print_with_timestamp(f"Learning rate Gen: {get_lr(optimizer_G):.6f}")

        epoch_Loss_D = 0
        epoch_Loss_G = 0
        epoch_Loss_G_identity = 0
        epoch_Loss_G_GAN = 0
        epoch_Loss_G_cycle = 0
        step = 0

        for i, data in enumerate(dataloader):
            step += 1
            # get batch size data
            real_image_A = data["A"].to(device)
            real_image_B = data["B"].to(device)

            ##############################################
            # (1) Update G network: Generators A2B and B2A
            ##############################################

            # Set G_A and G_B's gradients to zero
            optimizer_G.zero_grad()

            with autocast():
                # Identity loss
                # G_B2A(A) should equal A if real A is fed
                identity_image_A = netG_B2A(real_image_A)
                loss_identity_A = (
                    identity_loss(identity_image_A, real_image_A)
                    * config.LAMBDA_IDENTITY
                )
                # G_A2B(B) should equal B if real B is fed
                identity_image_B = netG_A2B(real_image_B)
                loss_identity_B = (
                    identity_loss(identity_image_B, real_image_B)
                    * config.LAMBDA_IDENTITY
                )
                # GAN loss
                # GAN loss D_A(G_A(A))
                fake_image_A = netG_B2A(real_image_B)
                fake_output_A = netD_A(fake_image_A)

                loss_GAN_B2A = adversarial_loss(
                    fake_output_A, torch.ones_like(fake_output_A)
                )

                # GAN loss D_B(G_B(B))
                fake_image_B = netG_A2B(real_image_A)
                fake_output_B = netD_B(fake_image_B)
                loss_GAN_A2B = adversarial_loss(
                    fake_output_B, torch.ones_like(fake_output_B)
                )

                # Cycle loss
                recovered_image_A = netG_B2A(fake_image_B)
                loss_cycle_ABA = (
                    cycle_loss(recovered_image_A, real_image_A) * config.LAMBDA_CYCLE
                )
                recovered_image_B = netG_A2B(fake_image_A)
                loss_cycle_BAB = (
                    cycle_loss(recovered_image_B, real_image_B) * config.LAMBDA_CYCLE
                )
                # Combined loss and calculate gradients
                errG = (
                    loss_identity_A
                    + loss_identity_B
                    + loss_GAN_A2B
                    + loss_GAN_B2A
                    + loss_cycle_ABA
                    + loss_cycle_BAB
                )

            # Calculate gradients for G_A and G_B
            scaler.scale(errG).backward()
            # Update G_A and G_B's weights
            scaler.step(optimizer_G)
            scaler.update()

            ##############################################
            # (2) Update D network: Discriminator A
            ##############################################

            # Set D_A gradients to zero
            optimizer_D_A.zero_grad()

            with autocast():
                # Real A image loss
                real_output_A = netD_A(real_image_A)
                errD_real_A = adversarial_loss(
                    real_output_A, torch.ones_like(real_output_A)
                )

                # Fake A image loss
                fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
                fake_output_A = netD_A(fake_image_A.detach())
                errD_fake_A = adversarial_loss(
                    fake_output_A, torch.zeros_like(fake_output_A)
                )

                # Combined loss and calculate gradients
                errD_A = (errD_real_A + errD_fake_A) / 2

            # Calculate gradients for D_A
            scaler.scale(errD_A).backward()
            # Update D_A weights
            scaler.step(optimizer_D_A)
            scaler.update()

            ##############################################
            # (3) Update D network: Discriminator B
            ##############################################

            # Set D_B gradients to zero
            optimizer_D_B.zero_grad()

            with autocast():
                # Real B image loss
                real_output_B = netD_B(real_image_B)
                errD_real_B = adversarial_loss(
                    real_output_B, torch.ones_like(real_output_B)
                )

                # Fake B image loss
                fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
                fake_output_B = netD_B(fake_image_B.detach())
                errD_fake_B = adversarial_loss(
                    fake_output_B, torch.zeros_like(fake_output_B)
                )

                # Combined loss and calculate gradients
                errD_B = (errD_real_B + errD_fake_B) / 2

            # Calculate gradients for D_B
            scaler.scale(errD_B).backward()
            # Update D_B weights
            scaler.step(optimizer_D_B)
            scaler.update()

            # Log loss
            epoch_Loss_D += (errD_A + errD_B).item()
            epoch_Loss_G += errG.item()
            epoch_Loss_G_identity += (loss_identity_A + loss_identity_B).item()
            epoch_Loss_G_GAN += (loss_GAN_A2B + loss_GAN_B2A).item()
            epoch_Loss_G_cycle += (loss_cycle_ABA + loss_cycle_BAB).item()

        # Average epoch loss
        epoch_Loss_D /= step
        epoch_Loss_G /= step
        epoch_Loss_G_identity /= step
        epoch_Loss_G_GAN /= step
        epoch_Loss_G_cycle /= step
        Loss_D.append(epoch_Loss_D)
        Loss_G.append(epoch_Loss_G)
        Loss_G_identity.append(epoch_Loss_G_identity)
        Loss_G_GAN.append(epoch_Loss_G_GAN)
        Loss_G_cycle.append(epoch_Loss_G_cycle)

        # Print epoch loss
        print_with_timestamp(f"Loss D: {epoch_Loss_D:.4f}")
        print_with_timestamp(f"Loss G: {epoch_Loss_G:.4f}")
        print_with_timestamp(f"Loss G_identity: {epoch_Loss_G_identity:.4f}")
        print_with_timestamp(f"Loss G_GAN: {epoch_Loss_G_GAN:.4f}")
        print_with_timestamp(f"Loss G_cycle: {epoch_Loss_G_cycle:.4f}")

        # Log loss
        loss_types = [Loss_D, Loss_G, Loss_G_identity, Loss_G_GAN, Loss_G_cycle]
        loss_names = [
            "Loss_D",
            "Loss_G",
            "Loss_G_identity",
            "Loss_G_GAN",
            "Loss_G_cycle",
        ]
        for loss_type, loss_name in zip(loss_types, loss_names):
            losses = {epoch + 1: value for epoch, value in enumerate(loss_type)}
            losses_items = losses.items()
            losses_list = list(losses_items)
            losses_df = pd.DataFrame(losses_list, columns=["Epoch", f"{loss_name}"])
            losses_df.to_csv(join(experiment_dir, f"{loss_name}.csv"), index=False)
            # Plot loss
            plot_loss(losses, join(experiment_dir, f"{loss_name}.html"), show=False)

        # Plot discriminator and generator loss in one figure
        plot_disc_gen_loss(
            Loss_D, Loss_G_GAN, filename=join(experiment_dir, "Loss_D_G_GAN.html")
        )

        # Save intermediate results
        if epoch % config.PRINT_FREQUENCY == 0:
            # Get filenames
            filenames_real_image_A = data["ID_A"]
            filenames_real_image_B = data["ID_B"]

            # Create fakes
            fake_image_A = netG_B2A(real_image_B)
            fake_image_B = netG_A2B(real_image_A)

            # Detach
            fake_image_A = fake_image_A.cpu().detach().numpy()
            fake_image_B = fake_image_B.cpu().detach().numpy()

            # Remove color channel
            fake_image_A = fake_image_A.reshape(
                len(fake_image_A),
                fake_image_A.shape[2],
                fake_image_A.shape[3],
                fake_image_A.shape[4],
            )
            fake_image_B = fake_image_B.reshape(
                len(fake_image_B),
                fake_image_B.shape[2],
                fake_image_B.shape[3],
                fake_image_B.shape[4],
            )

            # Save fake images A (images that are generated to look like A)
            for image, name in zip(fake_image_A, filenames_real_image_B):
                # Read original image to obtain sitk metadata
                meta_image = sitk.ReadImage(join(config.DATA_DIR, "train/labels", name))
                image = inverse_transform(image, meta_image)
                # Save fake
                sitk.WriteImage(
                    image,
                    join(
                        experiment_dir,
                        "A",
                        f"fake_epoch_{epoch}_{name.split('.')[0]}.nii.gz",
                    ),
                )
                if config.SAVE_REAL_IMG:
                    # Save real
                    sitk.WriteImage(
                        meta_image,
                        join(
                            experiment_dir,
                            "B",
                            f"real_epoch_{epoch}_{name.split('.')[0]}.nii.gz",
                        ),
                    )

            # Save fake images B (images that are generated to look like B)
            for image, name in zip(fake_image_B, filenames_real_image_A):
                # Read original image to obtain sitk metadata
                meta_image = sitk.ReadImage(join(config.DATA_DIR, "train/images", name))
                image = inverse_transform(image, meta_image)
                # Save fake
                sitk.WriteImage(
                    image,
                    join(
                        experiment_dir,
                        "B",
                        f"fake_epoch_{epoch}_{name.split('.')[0]}.nii.gz",
                    ),
                )
                if config.SAVE_REAL_IMG:
                    # Save real
                    sitk.WriteImage(
                        meta_image,
                        join(
                            experiment_dir,
                            "A",
                            f"real_epoch_{epoch}_{name.split('.')[0]}.nii.gz",
                        ),
                    )

        # Do checkpointing
        if epoch % config.CHECKPOINT_FREQUENCY == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": netG_A2B.state_dict(),
                "optimizer_state_dict": optimizer_G.state_dict(),
                "scheduler_state_dict": lr_scheduler_G.state_dict(),
            }
            save_checkpoint(
                checkpoint,
                join(experiment_dir, "weights", f"netG_A2B_epoch_{epoch}.pth.tar"),
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": netG_B2A.state_dict(),
                "optimizer_state_dict": optimizer_G.state_dict(),
                "scheduler_state_dict": lr_scheduler_G.state_dict(),
            }
            save_checkpoint(
                checkpoint,
                join(experiment_dir, "weights", f"netG_B2A_epoch_{epoch}.pth.tar"),
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": netD_A.state_dict(),
                "optimizer_state_dict": optimizer_D_A.state_dict(),
                "scheduler_state_dict": lr_scheduler_D_A.state_dict(),
            }
            save_checkpoint(
                checkpoint,
                join(experiment_dir, "weights", f"netD_A_epoch_{epoch}.pth.tar"),
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": netD_B.state_dict(),
                "optimizer_state_dict": optimizer_D_B.state_dict(),
                "scheduler_state_dict": lr_scheduler_D_B.state_dict(),
            }
            save_checkpoint(
                checkpoint,
                join(experiment_dir, "weights", f"netD_B_epoch_{epoch}.pth.tar"),
            )

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Report epoch time
        epoch_end = perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_time = timedelta(seconds=epoch_time)
        print_with_timestamp(f"This epoch took {epoch_time.seconds} seconds")
        print()

    # Report run time
    end_time = perf_counter()
    run_time = end_time - start_time
    print_with_timestamp("FINISHED TRAINING")
    print_with_timestamp(f"Run time in hh:mm:ss.us: {timedelta(seconds=run_time)}")
    print_with_timestamp(f"Output saved in: {experiment_dir}")


if __name__ == "__main__":
    train()
