import argparse
import logging
import os

import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from tabulate import tabulate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LabelledDataset, UnlabelledDataset, EMADataLoader
from dataset import collate_fn_labelled, collate_fn_unlabelled, collate_fn_to_be_labelled

from crnn import CRNN
from resrnn import ResRNN
from vitt import ViTT
from vit_gru import ViTGRU
from ema import EMA

from metrics import get_audio_tagging_results, compute_metrics
from utils import median_filter, decode_timestamps, binarize, pred_to_time, train_test_split, encode_label


def get_args():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument("--model", type=str, choices=["crnn", "vitt", "vit_gru", "resrnn"])
    parser.add_argument("--pooling_type", type=str, choices=["avg", "max", "adaptive_avg", "adaptive_max"], default="max")
    # for dataset
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--augument", action="store_true", default=False)
    # for dataloader
    parser.add_argument("--batch_size", type=int, default=64)
    # for training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0x66ccff)
    parser.add_argument('--device', type=int, default=0)
    # for evaluation
    parser.add_argument('--time_ratio', type=float, default=0.02)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--window_size', type=int, default=1)
    # For output
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    return args


def train(model, ema, device, train_loader, criterion, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        iid_label_feature, iid_label_target = data[0]["features"].to(device), data[0]["targets"].to(device)
        iid_unlabel_feature = data[1]["features"].to(device)
        ood_unlabel_feature = data[2]["features"].to(device)
        # forward
        iid_label_pred_model = model(iid_label_feature)
        iid_unlabel_pred_model = model(iid_unlabel_feature)
        ood_unlabel_pred_model = model(ood_unlabel_feature)
        # ema forward
        iid_label_pred_ema = ema.ema_model(iid_label_feature)
        iid_unlabel_pred_ema = ema.ema_model(iid_unlabel_feature)
        ood_unlabel_pred_ema = ema.ema_model(ood_unlabel_feature)
        # loss
        loss = criterion(
            [iid_label_pred_model, iid_unlabel_pred_model, ood_unlabel_pred_model], 
            [iid_label_pred_ema, iid_unlabel_pred_ema, ood_unlabel_pred_ema],
            iid_label_target
        )
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        # log
        if (batch_idx + 1) % log_interval == 0:
            logging.info(f"Train Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item()}")
   
            
def validate(model, device, valid_loader, criterion, epoch, is_ema=False):
    model.eval()
    preds, targets = [], []
    loss_sum = 0
    with torch.no_grad():
        for batch in valid_loader:
            features = batch["features"].to(device)
            target = batch["targets"].to(device)
            # forward
            output = model(features)
            loss = criterion(output, target)
            loss_sum += loss.item()
            # save
            pred = torch.round(output["clip_prob"])
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
        # log
        loss_sum = loss_sum / len(valid_loader)
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        p, r, f1, _ = skmetrics.precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
        logging.info(
            f"Valid {'Model' if not is_ema else 'EMA'} Epoch: {epoch} Loss: {loss_sum}\n"
            f"Precision: {p:.4f} Recall: {r:.4f} F1: {f1:.4f}"
        )
    return loss_sum
            
            
def evaluate(
        model, 
        device, 
        test_loader, 
        output_dir, 
        label_data, 
        label_to_idx, 
        idx_to_label, 
        time_ratio=0.02, 
        threshold=0.5, 
        window_size=1
    ):
    model.eval()
    clip_probs, clip_targets = [], []
    frame_preds, clip_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            audio_ids = batch["audio_ids"]
            features = batch["features"].to(device)
            # forward
            output = model(features)
            frame_prob = output["time_prob"].cpu().numpy()
            clip_prob = output["clip_prob"].cpu().numpy()
            filtered_frame_pred = median_filter(frame_prob, window_size=window_size, threshold=threshold)
            frame_pred = decode_timestamps(idx_to_label, filtered_frame_pred)
            # iter each clip
            for idx in range(len(audio_ids)):
                audio_id = audio_ids[idx]
                clip_probs.append(clip_prob[idx])
                clip_target = label_data[label_data["filename"] == audio_id]["event_label"].unique()
                clip_targets.append(encode_label(clip_target, label_to_idx))
                # clip pred
                clip_pred = clip_prob[idx].reshape(1, -1)
                clip_pred = binarize(clip_pred, threshold)[0]
                clip_pred = [idx_to_label[i] for i, target in enumerate(clip_pred) if target == 1]
                for clip_label in clip_pred:
                    clip_preds.append({
                        "filename": audio_id,
                        "event_label": clip_label,
                        "probability": clip_prob[idx][label_to_idx[clip_label]]
                    })
                # frame pred
                for event_label, onset, offset in frame_pred[idx]:
                    frame_preds.append({
                        "filename": audio_id,
                        "event_label": event_label,
                        "onset": onset,
                        "offset": offset
                    })
    assert len(frame_preds) > 0, "Threshold is too high, no frame is detected."
    clip_pred_data = pd.DataFrame(clip_preds, columns=["filename", "event_label", "probability"])
    frame_pred_data = pd.DataFrame(frame_preds, columns=["filename", "event_label", "onset", "offset"])
    frame_pred_data = pred_to_time(frame_pred_data, ratio=time_ratio)
    frame_pred_data.to_csv(F"{output_dir}/predicitons.csv", index=False, sep="\t", float_format="%.3f")
    tagging_data = get_audio_tagging_results(label_data, clip_pred_data, label_to_idx)
    
    clip_targets = np.stack(clip_targets)
    clip_probs = np.stack(clip_probs)
    precision = skmetrics.average_precision_score(clip_targets, clip_probs, average=None)

    tagging_data.to_csv(F"{output_dir}/tagging.txt", index=False, sep="\t", float_format="%.3f")

    event_result, segment_result = compute_metrics(label_data, frame_pred_data)
    with open(f"{output_dir}/event.txt", "w") as f:
        f.write(str(event_result))
    with open(f"{output_dir}/segment.txt", "w") as f:
        f.write(str(segment_result))
        
    event_based_results = pd.DataFrame(event_result.results_class_wise_average_metrics()['f_measure'], index=['event_based'])
    segment_based_results = pd.DataFrame(segment_result.results_class_wise_average_metrics()['f_measure'], index=['segment_based'])
    result = pd.concat([event_based_results, segment_based_results])
    
    tagging_result = tagging_data.loc[tagging_data['label'] == 'macro'].values[0][1:]
    result.loc["tagging_based"] = list(tagging_result)
    result_table = tabulate(result, headers='keys', tablefmt='github')
    
    with open(f"{output_dir}/report.md", "w") as f:
        print(result_table, file=f)
        print(f"mAP: {np.mean(precision)}", file=f)
        
    logging.info(
        f"\n{result_table}\n"
        f"mAP: {np.mean(precision)}"
    )
        
    
def main(args):
    # prepare
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    logging.info(f"Using device: {device}")
    # data
    classinfo_filename = f"{args.data_path}/metadata/class_label_indices.txt"
    iid_label_label_filename = f"{args.data_path}/dev/label_weak.csv"
    iid_label_feature_filename = f"{args.data_path}/dev/feature_weak.csv"
    test_label_filename = f"{args.data_path}/eval/label.csv"
    test_feature_filename = f"{args.data_path}/eval/feature.csv"
    train_label, valid_label, label_to_idx, idx_to_label = train_test_split(
        classinfo_filename, iid_label_label_filename, test_size=args.test_size
    )
    test_label = pd.read_csv(test_label_filename, sep='\s+').convert_dtypes()
    iid_unlabel_feature_filename = f"{args.data_path}/dev/feature_unlabel_in_domain.csv"
    ood_unlabel_feature_filename = f"{args.data_path}/dev/feature_unlabel_out_of_domain.csv"
    # dataset
    iid_label_dataset = LabelledDataset(iid_label_feature_filename, train_label, label_to_idx)
    iid_unlabel_dataset = UnlabelledDataset(iid_unlabel_feature_filename)
    ood_unlabel_dataset = UnlabelledDataset(ood_unlabel_feature_filename)
    valid_dataset = LabelledDataset(iid_label_feature_filename, valid_label, label_to_idx)
    test_dataset = UnlabelledDataset(test_feature_filename)
    # dataloder
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_labelled)
    train_loader = EMADataLoader(
        [iid_label_dataset, iid_unlabel_dataset, ood_unlabel_dataset],
        args.batch_size,
        shuffle=True,
        collate_fns=[collate_fn_labelled, collate_fn_to_be_labelled, collate_fn_to_be_labelled]
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_labelled)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_unlabelled)
    # model
    if args.model == "crnn":
        model = CRNN(num_freq=valid_dataset.data_dim, num_class=len(label_to_idx), pooling_type=args.pooling_type).to(device)
    elif args.model == "vitt":
        model = ViTT(num_freq=valid_dataset.data_dim, num_class=len(label_to_idx), pooling_type=args.pooling_type).to(device)
    elif args.model == "vit_gru":
        model = ViTGRU(num_freq=valid_dataset.data_dim, num_class=len(label_to_idx), pooling_type=args.pooling_type).to(device)
    elif args.model == "resrnn":
        model = ResRNN(num_freq=valid_dataset.data_dim, num_class=len(label_to_idx)).to(device)
    else:
        raise NotImplementedError
    model_name = f"{args.save_dir}/{args.model}_{args.pooling_type}_{args.threshold}_ema_model.pth"
    ema_name = f"{args.save_dir}/{args.model}_{args.pooling_type}_{args.threshold}_ema_ema.pth"
    output_dir = f"{args.output_dir}/{args.model}_{args.threshold}_ema"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ema
    ema = EMA(model)
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor)
    # criterion
    def criterion(model_outputs, ema_outputs, target):
        class_loss = F.binary_cross_entropy_with_logits(model_outputs[0]["clip_prob"], target)
        ema_class_loss = F.binary_cross_entropy_with_logits(ema_outputs[0]["clip_prob"], target)
        residual_loss = 0
        consistency_loss = 0
        for model_output, ema_output in zip(model_outputs, ema_outputs):
            residual_loss += softmax_mse_loss(model_output, ema_output)
            consistency_loss += symmetric_mse_loss(model_output, ema_output)
        loss = class_loss + consistency_loss + residual_loss
        return loss
    
    def softmax_mse_loss(output1, output2):
        output1_clip = F.softmax(output1["clip_prob"], dim=1)
        output2_clip = F.softmax(output2["clip_prob"], dim=1)
        output1_time = F.softmax(output1["time_prob"], dim=1)
        output2_time = F.softmax(output2["time_prob"], dim=1)
        num_classes = output1_clip.shape[1]
        clip_loss = F.mse_loss(output1_clip, output2_clip, reduction="sum") / num_classes
        time_loss = F.mse_loss(output1_time, output2_time, reduction="sum") / num_classes
        return clip_loss + time_loss
    
    def symmetric_mse_loss(output1, output2):
        num_classes = output1["clip_prob"].shape[1]
        clip_loss = F.mse_loss(output1["clip_prob"], output2["clip_prob"], reduction="sum") / num_classes
        time_loss = F.mse_loss(output1["time_prob"], output2["time_prob"], reduction="sum") / num_classes
        return clip_loss + time_loss
    # train
    best_loss = 0x66ccff
    not_improve_count = 0
    for epoch in range(1, args.epochs + 1):
        train(model, ema, device, train_loader, criterion, optimizer, epoch, args.log_interval)
        val_loss = validate(ema, device, valid_loader, lambda output, target: F.binary_cross_entropy(output["clip_prob"], target), epoch, is_ema=False)
        _ = validate(ema.ema_model, device, valid_loader, lambda output, target: F.binary_cross_entropy(output["clip_prob"], target), epoch, is_ema=True)
        if val_loss < best_loss:
            best_loss = val_loss
            not_improve_count = 0
            torch.save(ema.model.state_dict(), model_name)
            torch.save(ema.ema_model.state_dict(), ema_name)
        else:
            not_improve_count += 1
        scheduler.step(val_loss)
        if not_improve_count == args.early_stop:
            break
    # test
    model.load_state_dict(torch.load(model_name))
    evaluate(model, device, test_loader, output_dir, test_label, label_to_idx, idx_to_label, args.time_ratio, args.threshold, args.window_size)
    
    
if __name__ == "__main__":
    args = get_args()
    main(args)
