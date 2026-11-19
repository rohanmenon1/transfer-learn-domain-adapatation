import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("runs", nargs="+")
args = p.parse_args()

models = [torch.load(f"models/{r}.pt") for r in args.runs]
torch.save({k: sum(m[k].float() for m in models) / len(models) for k in models[0]}, f"models/{args.out}.pt")
