import argparse
import numpy as np
from PIL import Image
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# use torch to generate a 224 by 224 randomly colored image
def generate_random_image(
    original_image_path, adversarial_image_path, output_image_path, size=(224, 224)
):

    img = Image.open(original_image_path).convert("RGB")
    img = img.crop((0, 0, size[0], size[1]))
    rgb = (
        torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                .view(224, 224, 3)
                .numpy()
                .astype("float32")
                / 255.0
            )
        )
        .permute(2, 0, 1)
        .contiguous()
    ).to(device)
    rgb.requires_grad_(False)

    # Generate a random image with the same size
    # adv_img = 0.1 * torch.randn((3, size[1], size[0]), device=device)
    # img = Image.fromarray(
    #     (adv_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # )
    # img.save("adv_image.png")
    adv_img = Image.open(adversarial_image_path).convert("RGB")
    adv_img = adv_img.crop((0, 0, size[0], size[1]))
    adv_img = (
        torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(adv_img.tobytes()))
                .view(224, 224, 3)
                .numpy()
                .astype("float32")
                / 255.0
            )
        )
        .permute(2, 0, 1)
        .contiguous()
    ).to(device)
    adv_img.requires_grad_(False)

    overlayed = (adv_img + rgb).clamp(0.0, 1.0)
    img = Image.fromarray(
        (overlayed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    img.save(output_image_path)
    print(f"Random image saved to {output_image_path}")


# function that parse args
def parse_args():
    parser = argparse.ArgumentParser(
        description="Applying adversarial overlay for Demo"
    )
    parser.add_argument(
        "--original",
        "-o",
        type=str,
        default="libero_spatial.png",
    )
    parser.add_argument(
        "--adversarial",
        "-adv",
        type=str,
        default="spatial_inner1000_outer50.png",
    )
    parser.add_argument(
        "--output",
        "-out",
        type=str,
        default="overlayed_adv.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_random_image(args.original, args.adversarial, args.output)
