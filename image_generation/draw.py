import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from vae_conv import VAEConv
from vae_linear import VAELinear
from vae_vit import VAEViT


@torch.no_grad()
def generate(model, z_dim, device):
    if z_dim == 1:
        # if z_dim is 1, we can use linspace to generate z
        z = torch.linspace(-5, 5, 256).reshape(-1, 1)
    elif z_dim == 2:
        # if z_dim is 2, we can use linspace to generate x and y, then use cartesian_prod to generate z
        # x is like [-5, 0, 5], y is like [-5, 0, 5], then z is like [(-5, -5), (-5, 0), ..., (5, 5)]
        x = torch.linspace(-5, 5, 16)
        y = torch.linspace(-5, 5, 16)
        z = torch.cartesian_prod(x, y)
    elif z_dim == 4:
        # if z_dim is 4, we can use linspace to generate a, b, c, d, then use cartesian_prod to generate z
        a = torch.linspace(-1, 1, 4)
        b = torch.linspace(-1, 1, 4)
        c = torch.linspace(-1, 1, 4)
        d = torch.linspace(-1, 1, 4)
        z = torch.cartesian_prod(a, b, c, d)
    else:
        raise ValueError(f"Unknown z_dim: {z_dim}")
    z = z.to(device)
    # call inference to generate x_hat
    x_hat = model.inference(z)
    return x_hat


def draw(x_hat, name):
    if not os.path.exists("images"):
        os.makedirs("images")
    # Generate a empty image with 256 patches
    image = np.zeros((28 * 16, 28 * 16))
    for i in range(16):
        for j in range(16):
            x_begin = i * 28
            x_end = (i + 1) * 28
            y_begin = j * 28
            y_end = (j + 1) * 28
            index = i * 16 + j
            image_item = x_hat[index, 0].cpu().numpy()
            image[x_begin:x_end, y_begin:y_end] = image_item
    plt.imsave(f"images/{name}", image, cmap="gray")


if __name__ == "__main__":
    device = f"cuda:1"
    models = os.listdir("models")
    for model in tqdm(models):
        "vae_vit_zdim4_lmbda_dynamic.pt"
        splices = model.split("_")
        model_type = splices[1]
        z_dim = int(splices[2].replace("zdim", ""))
        if model_type == "linear":
            vae = VAELinear(z_dim=z_dim)
        elif model_type == "conv":
            vae = VAEConv(z_dim=z_dim)
        elif model_type == "vit":
            vae = VAEViT(z_dim=z_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        # Load model
        vae.load_state_dict(torch.load(os.path.join("models", model)))
        vae = vae.to(device)
        vae.eval()
        # Generate x_hat
        x_hat = generate(vae, z_dim, device)
        # Draw x_hat
        draw(x_hat, model.replace(".pt", ".png"))
