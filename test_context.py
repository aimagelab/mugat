"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import argparse
import json
import os
import logging
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nougat import NougatModel
from nougat.metrics import compute_metrics
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import NougatDataset
from nougat.utils.device import move_to_device
from lightning_module import NougatDataPLModule
from enum import Enum


def test(args):
    pretrained_model = NougatModel.from_pretrained(args.checkpoint)
    pretrained_model = move_to_device(pretrained_model)

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    else:
        logging.warning("Results can not be saved. Please provide a -o/--save_path")
    predictions = []
    ground_truths = []
    metrics_on = defaultdict(list)
    metrics_op = defaultdict(list)
    metrics_neither = defaultdict(list)
    metrics_both = defaultdict(list)
    dataset = NougatDataset(
        dataset_path=args.dataset,
        nougat_model=pretrained_model,
        max_length=pretrained_model.config.max_length,
        split=args.split,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=args.shuffle,
        collate_fn=NougatDataPLModule.ignore_none_collate,
    )

    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        image_tensors, prev_image_tensors, next_image_tensors, decoder_input_ids, _ = sample
        if image_tensors is None:
            return
        if len(predictions) >= args.num_samples:
            break
        if(prev_image_tensors.max() == 0) or (next_image_tensors.max() == 0):
            continue
    
        for setting in ["op", "on", "neither", "both"]:
            ground_truth = pretrained_model.decoder.tokenizer.batch_decode(
                decoder_input_ids, skip_special_tokens=True
            )
            
            pretrained_model = pretrained_model.float()
            if setting == "op":
                outputs = pretrained_model.inference(
                    image_tensors=image_tensors,
                    prev_image_tensors=prev_image_tensors,
                    next_image_tensors=torch.zeros(1),
                    return_attentions=False,
                )["predictions"]
            elif setting == "on":
                outputs = pretrained_model.inference(
                    image_tensors=image_tensors,
                    prev_image_tensors=torch.zeros(1),
                    next_image_tensors=next_image_tensors,
                    return_attentions=False,
                )["predictions"]
            elif setting == "neither":
                outputs = pretrained_model.inference(
                    image_tensors=image_tensors,
                    prev_image_tensors=torch.zeros(1),
                    next_image_tensors=torch.zeros(1),
                    return_attentions=False,
                )["predictions"]
            elif setting == "both":
                outputs = pretrained_model.inference(
                    image_tensors=image_tensors,
                    prev_image_tensors=prev_image_tensors,
                    next_image_tensors=next_image_tensors,
                    return_attentions=False,
                )["predictions"]
            predictions.extend(outputs)
            ground_truths.extend(ground_truth)
            with Pool(args.batch_size) as p:
                _metrics = p.starmap(compute_metrics, iterable=zip(outputs, ground_truth))
                for m in _metrics:
                    for key, value in m.items():
                        if setting == "op":
                            metrics_op[key].append(value)
                        elif setting == "on":
                            metrics_on[key].append(value)
                        elif setting == "neither":
                            metrics_neither[key].append(value)
                        elif setting == "both":
                            metrics_both[key].append(value)    

                if setting == "op":
                    print("op")
                    print({key: sum(values) / len(values) for key, values in metrics_op.items()})
                elif setting == "on":
                    print("on")
                    print({key: sum(values) / len(values) for key, values in metrics_on.items()})
                elif setting == "neither":
                    print("neither")
                    print({key: sum(values) / len(values) for key, values in metrics_neither.items()})
                elif setting == "both":
                    print("both")
                    print({key: sum(values) / len(values) for key, values in metrics_both.items()})

    scores = {}

    for metric, vals in metrics_neither.items():
        scores[f"{metric}_accuracy"] = np.mean(vals)
    try:
        print(
            f"Neither: Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except:
        pass
    if args.save_path:
        with open(f"{args.save_path}_neither.jsonl", "w") as f:
            json.dump(scores, f)

    scores = {}
    for metric, vals in metrics_on.items():
        scores[f"{metric}_accuracy"] = np.mean(vals)
    try:
        print(
            f"Only Next: Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except:
        pass
    if args.save_path:
        with open(f"{args.save_path}_on.jsonl", "w") as f:
            json.dump(scores, f)

    scores = {}
    for metric, vals in metrics_op.items():
        scores[f"{metric}_accuracy"] = np.mean(vals)
    try:
        print(
            f"Only Previous: Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except:
        pass

    scores = {}
    if args.save_path:
        with open(f"{args.save_path}_op.jsonl", "w") as f:
            json.dump(scores, f)

    for metric, vals in metrics_both.items():
        scores[f"{metric}_accuracy"] = np.mean(vals)
    try:
        print(
            f"Both: Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except:
        pass
    if args.save_path:
        with open(f"{args.save_path}_both.jsonl", "w") as f:
            json.dump(scores, f)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--save_path", "-o", type=str, default=None, help="json file to save results to"
    )
    parser.add_argument("--num_samples", "-N", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    args, left_argv = parser.parse_known_args()
    args.checkpoint = get_checkpoint(args.checkpoint)

    predictions = test(args)
