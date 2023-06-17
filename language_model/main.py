import argparse
import logging
import os

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Corpus, CorpusDataLoader
from bert import BertModel
from feedforward import FeedForward
from transformer import Transformer
from rnn import RNN


def get_args():
    parser = argparse.ArgumentParser()
    # For model
    parser.add_argument("--model", type=str, choices=["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer", "FeedForward", "BertModel"], default="LSTM")
    parser.add_argument("--num_embeddings", type=int, default=256),
    parser.add_argument("--num_hidden", type=int, default=256),
    parser.add_argument("--num_layers", type=int, default=2),
    parser.add_argument("--num_heads", type=int, default=2),
    parser.add_argument("--dropout", type=float, default=0.2),
    parser.add_argument("--tie_weights", action="store_true"),
    # For data
    parser.add_argument("--data_path", type=str, default="./data/gigaspeech")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--bptt", type=int, default=35)
    # For training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=10)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0x66ccff)
    parser.add_argument("--save_path", type=str, default="./models")
    parser.add_argument("--log_path", type=str, default="./logs")

    args = parser.parse_args()
    return args


def loss(output, target):
    return F.nll_loss(output, target)


def PPL(loss_value):
    return torch.exp(loss_value)


def calulate_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, device, train_loader, lr, clip, epoch, log_interval, writer):
    model.train()
    total_loss = 0
    if isinstance(model, RNN):
        # Initialize hidden state
        hidden = model.init_hidden(train_loader.batch_size)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        if isinstance(model, RNN):
            output, hidden = model(data, hidden)
        else:
            output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        # Clip gradient
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Update parameters
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        # Write to tensorboard
        writer.add_scalar("train/loss", loss_value.item(), (epoch - 1) * len(train_loader) + batch_idx)
        # Log
        total_loss += loss_value
        if (batch_idx + 1) % log_interval == 0:
            current_loss = total_loss / log_interval
            PPL_value = PPL(current_loss)
            total_loss = 0
            logging.info(f"Train Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {current_loss.item():.4f} PPL: {PPL_value.item():.4f}")


def evalute(model, device, data_loader, epoch, writer):
    model.eval()
    loss_value = 0
    if isinstance(model, RNN):
        hidden = model.init_hidden(data_loader.batch_size)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            if isinstance(model, RNN):
                output, hidden = model(data, hidden)
            else:
                output = model(data)
            loss_current = loss(output, target)
            loss_value += loss_current * len(data)
            # Write to tensorboard
            writer.add_scalar("eval/loss", loss_current.item(), (epoch - 1) * len(data_loader) + batch_idx)
    loss_value /= (len(data_loader.data) - 1)
    PPL_value = PPL(loss_value)
    logging.info(f"Eval Epoch: {epoch} Loss: {loss_value.item():.4f} PPL: {PPL_value.item():.4f}")
    return loss_value, PPL_value


def main(args):
    # Set up
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    logging.info(f"Using device: {device}")
    # Create dataset
    corpus = Corpus(args.data_path)
    logging.info(f"Vocabulary size: {len(corpus.dictionary)}")
    train_loader = CorpusDataLoader(corpus, args.batch_size, args.bptt, dataset_type="train")
    valid_loader = CorpusDataLoader(corpus, args.batch_size, args.bptt, dataset_type="valid")
    test_loader = CorpusDataLoader(corpus, args.batch_size, args.bptt, dataset_type="test")
    # Create summary writer
    writer = SummaryWriter(args.log_path)
    # Create model
    num_tokens = len(corpus.dictionary)
    if args.model == "Transformer":
        model = Transformer(
            num_tokens=num_tokens,
            num_embeddings=args.num_embeddings,
            num_heads=args.num_heads,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
            is_tie_weights=args.tie_weights,
        ).to(device)
        model_name = f"transformer_embed{args.num_embeddings}_hidden{args.num_hidden}_layers{args.num_layers}_heads{args.num_heads}_dropout{args.dropout}_tie{int(args.tie_weights)}.pt"
    elif args.model == "BertModel":
        model = BertModel(
            num_tokens=num_tokens,
            num_embeddings=args.num_embeddings,
            num_heads=args.num_heads,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
            is_tie_weights=args.tie_weights,
        ).to(device)
        model_name = f"bert_embed{args.num_embeddings}_hidden{args.num_hidden}_layers{args.num_layers}_dropout{args.dropout}_tie{int(args.tie_weights)}.pt"
    elif args.model == "FeedForward":
        model = FeedForward(
            num_tokens=num_tokens,
            num_embeddings=args.num_embeddings,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
            is_tie_weights=args.tie_weights,
        ).to(device)
        model_name = f"feedforward_embed{args.num_embeddings}_hidden{args.num_hidden}_layers{args.num_layers}_dropout{args.dropout}_tie{int(args.tie_weights)}.pt"
    else:
        model = RNN(
            rnn_type=args.model,
            num_tokens=num_tokens,
            num_embeddings=args.num_embeddings,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
            is_tie_weights=args.tie_weights
        ).to(device)
        model_name = f"{args.model.lower()}_embed{args.num_embeddings}_hidden{args.num_hidden}_layers{args.num_layers}_dropout{args.dropout}_tie{int(args.tie_weights)}.pt"
    logging.info(f"Model size: {calulate_model_size(model)/1e6:.2f}M")
    # Train
    loss_last = 0x66ccff
    PPL_last = 0x66ccff
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, args.lr, args.clip, epoch, args.log_interval, writer)
        loss_current, PPL_current = evalute(model, device, valid_loader, epoch, writer)
        if loss_current < loss_last:
            torch.save(model.state_dict(), f"{args.save_path}/{model_name}")
            if PPL_last - PPL_current < 1e-1:
                break
            loss_last = loss_current
            PPL_last = PPL_current
        else:
            args.lr /= 4.0
    # Test
    model.load_state_dict(torch.load(f"{args.save_path}/{model_name}"))
    loss_test, PPL_test = evalute(model, device, test_loader, -1, writer)
    logging.info(f"Test Loss: {loss_test.item():.4f} PPL: {PPL_test.item():.4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)
