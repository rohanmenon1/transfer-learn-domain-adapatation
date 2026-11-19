import json
import sys

from common import DOMAINS

runs = {r: json.load(open(f"results/{r}.json")) for r in sys.argv[1:]}
print("| run | " + " | ".join(DOMAINS) + " | epochs to 90% of final (in-domain) |")
print("|---|" + "---|" * (len(DOMAINS) + 1))
for r, d in runs.items():
    src, h = d["args"]["source"], d["history"]
    acc = d.get("final") or h[-1]
    to90 = next(x["epoch"] for x in h if x[src] >= 0.9 * h[-1][src])
    print(f"| {r} | " + " | ".join(f"{100*acc[t]:.1f}" for t in DOMAINS) + f" | {to90} |")
