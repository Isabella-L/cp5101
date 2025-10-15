import numpy as np
from PIL import Image
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# use torch to generate a 224 by 224 randomly colored image
def generate_random_image(size=(224, 224)):

    img = Image.open("overlayed_og.png").convert("RGB")
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
    adv_img = Image.open("prj_adv.png").convert("RGB")
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
    img.save("overlayed_adv.png")
    print(f"Random image saved to overlayed_adv.png")


if __name__ == "__main__":
    generate_random_image()
