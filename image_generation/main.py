import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ImageGenerationDataset
from vae_conv import VAEConv
from vae_linear import VAELinear
from vae_vit import VAEViT


def get_args():
    parser = argparse.ArgumentParser()
    # For model
    parser.add_argument("--model", type=str, choices=["vae_conv", "vae_linear", "vae_vit"])
    parser.add_argument("--z_dim", type=int, default=1)
    # For dataset
    parser.add_argument("--data_path", type=str, default="./data")
    # For dataloader
    parser.add_argument("--batch_size", type=int, default=64)
    # For training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0x66ccff)
    parser.add_argument("--save_path", type=str, default="./models")
    # For loss
    parser.add_argument("--dynamic_lmbda", action="store_true")
    parser.add_argument("--lmbda", type=float, default=1.0)
    # For lr scheduler
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    args = parser.parse_args()
    return args


def recon_loss(x, x_hat):
    """
    L = E_{z\sim Q}[log P(X|z)]
    """
    return F.binary_cross_entropy(x_hat, x, reduction="sum")


def kl_divergence_loss(mu, log_var):
    """
    L = -D_{KL}(Q(z|X)||P(z))
    """
    return -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))


def loss(x, x_hat, mu, log_var, lmbda=1.0):
    """
    L = E_{z\sim Q}[log P(X|z)] - \lambda D_{KL}(Q(z|X)||P(z))
    """
    recon = recon_loss(x, x_hat)
    kl = kl_divergence_loss(mu, log_var)
    return recon + lmbda * kl


def train(model, device, train_loader, optimizer, epoch, log_interval, lmbda):
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        x_hat, mu, log_var = model(x)
        loss_value = loss(x, x_hat, mu, log_var, lmbda)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            logging.info(f"Train Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {loss_value.item() / x.size(0)}")


def test(model, device, test_loader, epoch):
    model.eval()
    recon_loss_value = 0
    kl_divergence_loss_value = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            recon_loss_value += recon_loss(x, x_hat)
            kl_divergence_loss_value += kl_divergence_loss(mu, log_var)
    recon_loss_value /= len(test_loader.dataset)
    kl_divergence_loss_value /= len(test_loader.dataset)
    logging.info(f"Test Epoch: {epoch} Recon Loss: {recon_loss_value.item()} KL Divergence Loss: {kl_divergence_loss_value.item()}")
    return recon_loss_value, kl_divergence_loss_value


def main(args):
    # Set up
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    logging.info(f"Using device: {device}")
    # Create dataset and dataloader
    train_dataset = ImageGenerationDataset(is_training=True, data_path=args.data_path)
    test_dataset = ImageGenerationDataset(is_training=False, data_path=args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Create model, optimizer and scheduler
    if args.model == "vae_conv":
        model = VAEConv(z_dim=args.z_dim).to(device)
    elif args.model == "vae_linear":
        model = VAELinear(z_dim=args.z_dim).to(device)
    elif args.model == "vae_vit":
        model = VAEViT(z_dim=args.z_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # Set loss value for early stopping
    recon_loss_last = 0x66ccff
    for epoch in range(1, args.epochs + 1):
        # Set lambda value
        if args.dynamic_lmbda:
            if epoch == 1:
                lmbda = 0.0
            else:
                lmbda = min(1.0, lmbda + 0.1)
            model_name = f"{args.save_path}/{args.model}_zdim{args.z_dim}_lmbda_dynamic.pt"
        else:
            lmbda = args.lmbda
            model_name = f"{args.save_path}/{args.model}_zdim{args.z_dim}_lmbda{lmbda}.pt"
        train(model, device, train_loader, optimizer, epoch, args.log_interval, lmbda)
        recon_loss_current, kl_divergence_loss_value = test(model, device, test_loader, epoch)
        scheduler.step()
        # Save model if loss value is smaller than last time (early stopping)
        if recon_loss_current < recon_loss_last:
            torch.save(model.state_dict(), model_name)
            if recon_loss_last - recon_loss_current < 1e-1:
                break
            recon_loss_last = recon_loss_current


if __name__ == "__main__":
    args = get_args()
    main(args)
