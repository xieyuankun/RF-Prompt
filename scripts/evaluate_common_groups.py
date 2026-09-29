#!/usr/bin/env python3
"""Regroup final scores into the four RAMI groups used for Table 2."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_metrics import compute_eer


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def metric(rows):
    labels = np.asarray([int(row['label']) for row in rows])
    scores = np.asarray([float(row['score_real']) for row in rows])
    eer, threshold = compute_eer(scores[labels == 0], scores[labels == 1])
    return {
        'samples': int(len(rows)),
        'eer_percent': float(100.0 * eer),
        'threshold': float(threshold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--rami_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    score_rows = read_rows(args.scores)
    score_by_id = {row['utt_id']: row for row in score_rows}
    if len(score_by_id) != len(score_rows):
        raise ValueError('Score file contains duplicate utterance IDs')
    manifests = sorted(args.rami_root.glob('*/eval.csv'))
    if len(manifests) != 4:
        raise ValueError(f'Expected four RAMI eval manifests, found {len(manifests)}')

    groups = {}
    used = []
    for manifest in manifests:
        source = read_rows(manifest)
        ids = [row['utt_id'] for row in source]
        missing = sorted(set(ids) - set(score_by_id))
        if missing:
            raise ValueError(f'{manifest} has {len(missing)} IDs without scores')
        rows = [score_by_id[utterance_id] for utterance_id in ids]
        expected_labels = [int(row['label']) for row in source]
        if expected_labels != [int(row['label']) for row in rows]:
            raise ValueError(f'Label mismatch while regrouping {manifest}')
        groups[manifest.parent.name] = metric(rows)
        used.extend(ids)
    if len(used) != len(set(used)) or set(used) != set(score_by_id):
        raise ValueError('RAMI groups must partition the complete score pool')

    result = {
        'common_groups': groups,
        'common_average_eer': float(np.mean([
            value['eer_percent'] for value in groups.values()
        ])),
        'pooled': metric(score_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
