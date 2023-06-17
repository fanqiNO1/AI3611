"""
parser.add_argument("--num_embeddings", type=int, default=256),
parser.add_argument("--num_hidden", type=int, default=256),
parser.add_argument("--num_layers", type=int, default=2),
parser.add_argument("--num_heads", type=int, default=2),
parser.add_argument("--dropout", type=float, default=0.2),
parser.add_argument("--tie_weights", action="store_true"),
"""

import sys
import os


def main(target, device):
    models = ["Transformer", "FeedForward"]
    if target == "num_embedding" or target == "num_hidden":
        target_range = [128 * i for i in range(1, 9)]
        name = "embed" if target == "num_embedding" else "hidden"
    elif target == "num_layers":
        target_range = [i for i in range(1, 9)]
        name = "layers"
    elif target == "num_heads":
        target_range = [2 ** i for i in range(1, 9)]
        name = "heads"
        models = ["Transformer"]
    elif target == "dropout":
        target_range = [0.1 * i for i in range(1, 6)]
        name = "dropout"
    elif target == "tie_weights":
        target_range = [True, False]
        name = "tie"

    for model in models:
        for value in target_range:
            if target != "tie_weights":
                log = f"{model}_{name}_{value}.log"
                cmd = f"nohup python main.py --model {model} --{target} {value} --device {device} > train_logs/{log} 2>&1"
            else:
                log = f"{model}_{name}_{int(value)}.log"
                if value:
                    cmd = f"nohup python main.py --model {model} --{target} --device {device} > train_logs/{log} 2>&1"
                else:
                    cmd = f"nohup python main.py --model {model} --device {device} > train_logs/{log} 2>&1"
            print(f"Running {cmd}")
            os.system(cmd)


if __name__ == "__main__":
    target = sys.argv[1]
    device = int(sys.argv[2])
    main(target, device)
