# Download and extract the user-selected DomainNet revision.
import json
from pathlib import Path, PurePosixPath


import dino_common as base

DOMAINS = tuple(base.DOMAINS)
DATASET = 'wltjr1007/DomainNet'
REVISION = 'ee20570ae7a29c51571e55a9a17983f7625295d6'


# Download the pinned six-domain dataset and preserve its split membership.
def prepare_dataset(data):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    data.mkdir(parents=True, exist_ok=True)
    marker = data / 'provenance.json'
    if marker.exists() and json.loads(marker.read_text())['revision'] == REVISION:
        return
    split_dir = data / 'splits'
    split_dir.mkdir(exist_ok=True)
    counts = {}
    for split, shards, expected in [('train', 3, 409832), ('test', 1, 176743)]:
        lists = {domain: [] for domain in DOMAINS}
        seen = set()
        for i in range(shards):
            filename = f'data/{split}-{i:05d}-of-{shards:05d}.parquet'
            print('Preparing', filename, flush=True)
            shard_path = hf_hub_download(DATASET, filename, repo_type='dataset', revision=REVISION,
                                         cache_dir=data / 'hub_cache')
            for batch in pq.ParquetFile(shard_path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    path = PurePosixPath(row['image_path'])
                    if path.is_absolute() or len(path.parts) < 2 or '..' in path.parts:
                        raise ValueError(f'Invalid image path: {path}')
                    domain = path.parts[0]
                    name = path.as_posix()
                    if domain not in lists or name in seen:
                        raise ValueError(f'Unexpected domain or duplicate path: {name}')
                    seen.add(name)
                    label = int(row['label'])
                    if not 0 <= label < base.CLASSES:
                        raise ValueError(f'Invalid label for {name}')
                    image = data.joinpath(*path.parts)
                    image.parent.mkdir(parents=True, exist_ok=True)
                    payload = row['image']['bytes']
                    if not payload:
                        raise ValueError(f'Missing image bytes: {name}')
                    if not image.exists() or image.stat().st_size != len(payload):
                        image.write_bytes(payload)
                    lists[domain].append(f'{name} {label}\n')
        if len(seen) != expected or any(not rows for rows in lists.values()):
            raise ValueError(f'Incomplete {split} split: {len(seen)} rows')
        for domain, lines in lists.items():
            (split_dir / f'{domain}_{split}.txt').write_text(''.join(lines), encoding='utf-8')
            counts[f'{domain}_{split}'] = len(lines)
    base.write_json(marker, {'repo': DATASET, 'revision': REVISION, 'counts': counts})

