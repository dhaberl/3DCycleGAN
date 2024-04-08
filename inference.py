import os
import sys
from glob import glob
from os.path import join

import numpy as np
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import AddChannel, Compose, EnsureType, LoadImage, ScaleIntensity
from natsort import natsorted

from inf_batch_dataset import InferenceBatchDataset
from model import Generator


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


def predict_from_folder():
    """
    Predicts images from input folder
    """

    # Set directories
    img_dir = "input_dir/"  # Path to directory containing images as nifti
    output_dir = "output_dir/"  # Path to where output will be saved
    model_dir = "model_dir/"  # weights/ folder of training output

    # Parameters
    model_checkpoint = 1000  # Checkoint from which the model will be loaded
    direction = "AtoB"  # either "AtoB" or "BtoA"; A = images, B = labels (see dataset)

    save_inferred_images = True  # Save images as nifti

    # Hyperparameters for sliding window inference
    run_sliding_window = True
    roi_size = (128, 128, 64)
    sw_batch_size = 2
    mode = "gaussian"
    overlap = 0.85  # greater .75 recommended

    # Enable builtin hardware optimization
    cudnn.benchmark = True

    # Load files
    img_paths = natsorted(glob(join(img_dir, "*.nii.gz")))

    # Transforms
    transforms = Compose(
        [
            LoadImage(image_only=True),
            AddChannel(),
            EnsureType(),
        ]
    )

    # Dataset
    inference_ds = InferenceBatchDataset(img_paths, transforms=transforms)

    # Dataloader
    inference_loader = DataLoader(
        inference_ds, batch_size=1, num_workers=32
    )  # batch_size must be 1 for sliding window inference!

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    netG_A2B = Generator(in_channels=1, num_residuals=9).to(device)
    netG_B2A = Generator(in_channels=1, num_residuals=9).to(device)

    # Use all available GPUs
    netG_A2B = nn.DataParallel(netG_A2B).to(device)
    netG_B2A = nn.DataParallel(netG_B2A).to(device)

    # Load state dicts
    print(f"Loading model from checkpoint: {model_checkpoint}")
    checkpoint_A2B = torch.load(
        join(model_dir, f"netG_A2B_epoch_{model_checkpoint}.pth.tar")
    )
    netG_A2B.load_state_dict(checkpoint_A2B["model_state_dict"])
    checkpoint_B2A = torch.load(
        join(model_dir, f"netG_B2A_epoch_{model_checkpoint}.pth.tar")
    )
    netG_B2A.load_state_dict(checkpoint_B2A["model_state_dict"])

    # Set model mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Make output directory
    output_dir = join(output_dir, f"inference_{model_checkpoint}epoch")
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    print("Running inference")
    print(f"Sliding window inference: {run_sliding_window}")
    with torch.no_grad():
        for index, batch in enumerate(inference_loader):
            uid = batch[1][0]
            print(f"Image {index+1}/{len(inference_loader)}: {uid}")

            # Assign batch
            original_image = batch[0].to(device)

            # Store original min and max intensity values
            orig_min = original_image.min().item()
            orig_max = original_image.max().item()
            print(f"Original intensity min: {orig_min}")
            print(f"Original intensity max: {orig_max}")

            # Scale [-1, +1]
            print("Normalizing original image to [-1, +1] for inference")
            scale_intensity_transform = ScaleIntensity(minv=-1, maxv=1)
            original_image = scale_intensity_transform(original_image)
            print(f"Normalized intensity min: {original_image.min().item()}")
            print(f"Normalized intensity max: {original_image.max().item()}")

            if direction == "AtoB":
                print("Transforming A (images) to B (labels)")
                if run_sliding_window:
                    img = sliding_window_inference(
                        original_image,
                        roi_size,
                        sw_batch_size,
                        netG_A2B,
                        overlap=overlap,
                        mode=mode,
                        device=device,
                    )
                else:
                    img = netG_A2B(original_image)
            elif direction == "BtoA":
                print("Transforming B (labels) to A (images)")
                if run_sliding_window:
                    img = sliding_window_inference(
                        original_image,
                        roi_size,
                        sw_batch_size,
                        netG_B2A,
                        overlap=overlap,
                        mode=mode,
                        device=device,
                    )
                else:
                    img = netG_B2A(original_image)
            else:
                print("Invalid direction.")
                sys.exit()

            # Inverse rescaling: from [-1, +1]
            # to [original min intensity, original max intensity]
            print(
                "Inverse rescaling of inferred image from [-1, +1] to original intensity values"
            )
            img = img.cpu().detach().numpy()
            img = img[0, 0, :, :, :]
            img = inverse_rescaling(img, mina=orig_min, maxa=orig_max)
            print(f"Inferred intensity min: {img.min()}")
            print(f"Inferred intensity max: {img.max()}")

            # Save inferred images (fake images)
            if save_inferred_images:
                print("Saving fake...")
                # Read original image to get metadata
                filepath = batch[2][0]
                reference_image = sitk.ReadImage(filepath)
                # Convert inferred image to sitk object
                img = sitk.GetImageFromArray(np.transpose(img))
                # Copy metadata from original image to inferred image
                img.CopyInformation(reference_image)
                # Save inferred image
                sitk.WriteImage(img, join(output_dir, f"{uid}_fake.nii.gz"))

            print()


if __name__ == "__main__":
    predict_from_folder()
