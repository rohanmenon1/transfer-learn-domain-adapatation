import argparse
import json
from pathlib import Path

import torch

from common import NUM_CLASSES, accuracy, backbone, test_loader

p = argparse.ArgumentParser()
p.add_argument("run")
args = p.parse_args()

out = Path(f"results/{args.run}.json")
result = json.loads(out.read_text())
model = backbone(result["args"]["backbone"], NUM_CLASSES)
model.load_state_dict(torch.load(f"models/{args.run}.pt"))
result["final"] = {d: accuracy(model, test_loader(d)) for d in result["args"]["targets"]}
print("final", result["final"])
out.write_text(json.dumps(result, indent=2))
