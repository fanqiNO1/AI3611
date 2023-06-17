import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, log_loss, classification_report
import seaborn as sns

from dataset import SceneEmbeddingDataset
from adaptiveconcatbase import AdaptiveConcatBase
from cnnconcatlate import CNNConcatLate
from meanconcatbase import MeanConcatBase
from meanconcatearly import MeanConcatEarly
from meanconcatlate import MeanConcatLate
from meanconcatmoe import MeanConcatMoE
from single import SingleModal
from vitconcatlate import ViTConcatLate


KEYS = [
    'airport',
    'bus',
    'metro',
    'metro_station',
    'park',
    'public_square',
    'shopping_mall',
    'street_pedestrian',
    'street_traffic',
    'tram'
]


def get_args():
    parser = argparse.ArgumentParser()
    # For model
    parser.add_argument('--model', type=str, choices=['meanbase', 'meanearly', 'meanlate', 'meanmoe', 'adaptivebase', 'vitlate', 'cnnlate', 'single'], default='baseline')
    parser.add_argument('--modal', type=str, choices=['audio', 'video'], default='audio')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    # For data
    parser.add_argument('--audio_path', type=str, default='data/feature/audio_features_data')
    parser.add_argument('--video_path', type=str, default='data/feature/video_features_data')
    parser.add_argument('--batch_size', type=int, default=64)
    # For training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=7)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0x66ccff)
    parser.add_argument('--device', type=int, default=0)
    # For output
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--output_dir', type=str, default='outputs')

    args = parser.parse_args()
    return args


def train(model, device, train_loader, criterion, optimizer, epoch, log_interval):
    model.train()
    loss_sum = 0
    for batch_idx, data in enumerate(train_loader):
        audio_feature = data['audio_feature'].to(device)
        video_feature = data['video_feature'].to(device)
        target = data['target'].to(device)
        # forward
        if isinstance(model, SingleModal):
            if model.modal == 'audio':
                output = model(audio_feature)
            elif model.modal == 'video':
                output = model(video_feature)
        else:
            output = model(audio_feature, video_feature)
        loss = criterion(output, target)
        loss_sum += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        if (batch_idx + 1) % log_interval == 0:
            logging.info(f"Train Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item()}")
    loss_sum /= len(train_loader)
    return loss_sum


def validate(model, device, valid_loader, criterion, epoch):
    model.eval()
    preds, targets = [], []
    loss_sum = 0
    with torch.no_grad():
        for data in valid_loader:
            audio_feature = data['audio_feature'].to(device)
            video_feature = data['video_feature'].to(device)
            target = data['target'].to(device)
            # forward
            if isinstance(model, SingleModal):
                if model.modal == 'audio':
                    output = model(audio_feature)
                elif model.modal == 'video':
                    output = model(video_feature)
            else:
                output = model(audio_feature, video_feature)
            # metrics
            loss = criterion(output, target)
            loss_sum += loss.item()
            pred = torch.argmax(output, dim=1)
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())

        loss_sum /= len(valid_loader)
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        acc = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average=None)
        min_precision, max_precision, avg_precision = np.min(precision), np.max(precision), np.mean(precision)
        recall = recall_score(targets, preds, average=None)
        min_recall, max_recall, avg_recall = np.min(recall), np.max(recall), np.mean(recall)
        f1 = f1_score(targets, preds, average=None)
        min_f1, max_f1, avg_f1 = np.min(f1), np.max(f1), np.mean(f1)
        logging.info(
            f"Valid Epoch: {epoch} Loss: {loss_sum} Acc: {acc}\n"
            f" Precision: Min: {min_precision} Max: {max_precision} Avg: {avg_precision}\n"
            f" Recall: Min: {min_recall} Max: {max_recall} Avg: {avg_recall}\n"
            f" F1: Min: {min_f1} Max: {max_f1} Avg: {avg_f1}"
        )

    return loss_sum


def evaluate(model, device, test_loader, output_dir):
    model.eval()
    files, probs, preds, targets = [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            file = data['file']
            audio_feature = data['audio_feature'].to(device)
            video_feature = data['video_feature'].to(device)
            target = data['target'].to(device)
            # forward
            if isinstance(model, SingleModal):
                if model.modal == 'audio':
                    output = model(audio_feature)
                elif model.modal == 'video':
                    output = model(video_feature)
            else:
                output = model(audio_feature, video_feature)
            # metrics
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)

            files.append(np.array(file))
            probs.append(prob.cpu().numpy())
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())

        files = np.concatenate(files, axis=0)
        probs = np.concatenate(probs, axis=0)
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        # save
        scenes_pred = [KEYS[pred] for pred in preds]
        scenes_target = [KEYS[target] for target in targets]
        pred_dict = {"aid": files, "scene_pred": scenes_pred, "scene_label": scenes_target}
        for idx, key in enumerate(KEYS):
            pred_dict[key] = probs[:, idx]
        pd.DataFrame(pred_dict).to_csv(f"{output_dir}/prediction.csv", index=False, sep="\t", float_format="%.3f")
        # confusion matrix
        cm = confusion_matrix(targets, preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(cm, index=KEYS, columns=KEYS)
        plt.clf()
        plt.figure(figsize=(15, 12))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(f"{output_dir}/cm.png")
        # classification report
        writer = open(f"{output_dir}/report.txt", "w")
        writer.write(classification_report(targets, preds, target_names=KEYS))
        acc = accuracy_score(targets, preds)
        writer.write(f"Accuracy: {acc}\n")
        precision = precision_score(targets, preds, average=None)
        min_precision, max_precision, avg_precision = np.min(precision), np.max(precision), np.mean(precision)
        writer.write(f"Precision: Min: {min_precision} Max: {max_precision} Avg: {avg_precision}\n")
        recall = recall_score(targets, preds, average=None)
        min_recall, max_recall, avg_recall = np.min(recall), np.max(recall), np.mean(recall)
        writer.write(f"Recall: Min: {min_recall} Max: {max_recall} Avg: {avg_recall}\n")
        f1 = f1_score(targets, preds, average=None)
        min_f1, max_f1, avg_f1 = np.min(f1), np.max(f1), np.mean(f1)
        writer.write(f"F1: Min: {min_f1} Max: {max_f1} Avg: {avg_f1}\n")
        logloss = log_loss(targets, probs)
        writer.write(f"Logloss: {logloss}")
        writer.close()
        

def main(args):
    # prepare
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    logging.info(f"Using device: {device}")
    # data
    # load mean and std
    mean_std_file = "global_mean_std.npz"
    mean_std_audio = np.load(f"{args.audio_path}/{mean_std_file}")
    mean_audio, std_audio = mean_std_audio['global_mean'], mean_std_audio['global_std']
    mean_std_video = np.load(f"{args.video_path}/{mean_std_file}")
    mean_video, std_video = mean_std_video['global_mean'], mean_std_video['global_std']
    # build transform
    audio_transform = lambda x: (x - mean_audio) / std_audio
    video_transform = lambda x: (x - mean_video) / std_video
    # build dataset
    train_file, val_file, test_file = "train.hdf5", "val.hdf5", "test.hdf5"
    train_dataset = SceneEmbeddingDataset(
        f"{args.audio_path}/{train_file}", f"{args.video_path}/{train_file}",
        audio_transform=audio_transform, video_transform=video_transform
    )
    val_dataset = SceneEmbeddingDataset(
        f"{args.audio_path}/{val_file}", f"{args.video_path}/{val_file}",
        audio_transform=audio_transform, video_transform=video_transform
    )
    test_dataset = SceneEmbeddingDataset(
        f"{args.audio_path}/{test_file}", f"{args.video_path}/{test_file}",
        audio_transform=audio_transform, video_transform=video_transform
    )
    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # model
    if args.model == 'meanbase':
        model = MeanConcatBase(512, 512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = f"MeanConcatBase_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    elif args.model == 'meanearly':
        model = MeanConcatEarly(512, 512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = f"MeanConcatEarly_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    elif args.model == 'meanlate':
        model = MeanConcatLate(512, 512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = f"MeanConcatLate_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    elif args.model == 'meanmoe':
        model = MeanConcatMoE(512, 512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = f"MeanConcatMoE_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    elif args.model == 'adaptivebase':
        model = AdaptiveConcatBase(512, 512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = f"AdaptiveConcatBase_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    elif args.model == 'vitlate':
        model = ViTConcatLate(512, 512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = f"ViTConcatLate_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    elif args.model == 'cnnlate':
        model = CNNConcatLate(512, 512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = f"CNNConcatLate_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    elif args.model == "single":
        model = SingleModal(512, num_classes=len(KEYS), hidden_dim=args.hidden_dim, dropout=args.dropout, modal=args.modal)
        model_name = f"SingleModal_{args.modal}_hidden{args.hidden_dim}_dropout{args.dropout}.pt"
    else:
        raise NotImplementedError
    model.to(device)
    output_dir = f"{args.output_dir}/{model_name.replace('.pt', '')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor)
    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    # train
    best_loss = 0x66ccff
    not_improve_epochs = 0
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs):
        train_loss = train(model, device, train_loader, criterion, optimizer, epoch, args.log_interval)
        val_loss = validate(model, device, val_loader, criterion, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            not_improve_epochs = 0
            torch.save(model.state_dict(), f"{args.save_dir}/{model_name}")
        else:
            not_improve_epochs += 1
        scheduler.step(val_loss)
        if not_improve_epochs == args.early_stop:
            break
    # test
    model.load_state_dict(torch.load(f"{args.save_dir}/{model_name}"))
    evaluate(model, device, test_loader, output_dir)
    # draw loss curve
    plt.clf()
    min_val_loss_index = np.argmin(val_losses)
    min_val_loss = np.min(val_losses)
    plt.plot(train_losses, "r")
    plt.plot(val_losses, "b")
    plt.axvline(x=min_val_loss_index, color="k", linestyle="--")
    plt.plot(min_val_loss_index, min_val_loss, "r*")
    plt.title(f"Model {model_name} loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.savefig(f"{output_dir}/loss.png")


if __name__ == "__main__":
    args = get_args()
    main(args)
