#!/usr/bin/env python3
"""Construct the five paper protocols from one locked sample pool."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
import shutil

DOMAINS = ["asv19_la", "asvspoof5_track1", "codecfake", "atadd_track2_speech"]
VARIANTS = {
    "protocol_1_dataset_real_dataset_fake": ("a", "dataset"),
    "protocol_2_dataset_real_mechanism_fake": ("b", "dataset"),
    "protocol_3_source_matched_real_mechanism_fake": ("b", "source_matched"),
    "protocol_4_mixed_real_dataset_fake": ("a", "unchanged"),
    "protocol_5_rami": ("b", "unchanged"),
}
SPLITS = ["train", "dev", "eval"]

def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(rows):
    return Counter((r["utt_id"], r["audio_path"], r["label"]) for r in rows)

def semantic(rows):
    return Counter(tuple(sorted((k, v) for k, v in r.items() if k != "task_id")) for r in rows)

def apportion(total, weights):
    denominator = sum(weights)
    assert denominator > 0
    exact = [Fraction(total * w, denominator) for w in weights]
    counts = [int(x) for x in exact]
    order = sorted(range(4), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    assert sum(counts) == total
    assert all(n == 0 for n, w in zip(counts, weights) if w == 0)
    return counts

def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True,
        help="locked source containing master/, protocol_a/, and protocol_b/",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    assert not output.exists(), f"Refusing to overwrite {output}"
    source_hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    layouts = {layout: sorted(p for p in (source / f"protocol_{layout}").iterdir() if p.is_dir())
               for layout in ["a", "b"]}
    assert all(len(v) == 4 for v in layouts.values())
    prepared, source_pools, checks, count_rows = {}, {}, {}, []
    allocation_rows = []
    for split in SPLITS:
        original = {l: [read(d / f"{split}.csv")[1] for d in layouts[l]] for l in layouts}
        fields, master = read(source / "master" / f"{split}.csv")
        pool = [r for task in original["b"] for r in task]
        source_pools[split] = pool
        assert identity(pool) == identity(master)
        assert identity(pool) == identity([r for t in original["a"] for r in t])
        assert len({r["utt_id"] for r in pool}) == len(pool)
        assert len({r["audio_path"] for r in pool}) == len(pool)
        real_by_domain = {d: [r for r in pool if r["label"] == "0" and r["dataset"] == d] for d in DOMAINS}
        assert sum(map(len, real_by_domain.values())) == sum(r["label"] == "0" for r in pool)
        for variant, (layout, mode) in VARIANTS.items():
            tasks = original[layout]
            if mode == "unchanged":
                assigned = [[dict(r) for r in t] for t in tasks]
            else:
                assigned = [[dict(r) for r in t if r["label"] == "1"] for t in tasks]
                for domain_index, domain in enumerate(DOMAINS):
                    real = sorted(real_by_domain[domain], key=lambda r: hashlib.sha256(
                        f"2026|{split}|{domain}|{r['utt_id']}".encode()).hexdigest())
                    fake_weights = [sum(r["label"] == "1" and r["dataset"] == domain for r in t) for t in tasks]
                    counts = ([len(real) if i == domain_index else 0 for i in range(4)]
                              if mode == "dataset" else apportion(len(real), fake_weights))
                    cursor = 0
                    for i, n in enumerate(counts):
                        task_id = f"{layout.upper()}{i}"
                        for row in real[cursor:cursor+n]:
                            assigned[i].append(dict(row, task_id=task_id))
                        cursor += n
                        if mode == "source_matched":
                            allocation_rows.append(dict(split=split, task=task_id, domain=domain,
                                fake_count=fake_weights[i], real_count=n))
                    assert cursor == len(real)
                for i in range(4):
                    assigned[i].sort(key=lambda r: r["utt_id"])
            flattened = [r for task in assigned for r in task]
            assert identity(flattened) == identity(pool)
            # Only active task_id may change; every other source field is preserved.
            assert semantic(flattened) == semantic([r for t in tasks for r in t])
            for i, rows in enumerate(assigned):
                assert identity([r for r in rows if r["label"] == "1"]) == identity(
                    [r for r in tasks[i] if r["label"] == "1"])
                assert all(r["task_id"] == f"{layout.upper()}{i}" for r in rows)
                reals = Counter(r["dataset"] for r in rows if r["label"] == "0")
                fakes = Counter(r["dataset"] for r in rows if r["label"] == "1")
                assert sum(reals.values()) > 0 and sum(fakes.values()) > 0
                if mode == "source_matched":
                    assert set(reals) <= set(fakes)
                count_rows.append(dict(protocol=variant, split=split, task=f"{layout.upper()}{i}",
                    real=sum(reals.values()), fake=sum(fakes.values()), total=len(rows),
                    real_by_domain=dict(reals), fake_by_domain=dict(fakes)))
                prepared[(variant, split, i)] = (fields, rows, layouts[layout][i].name)
            checks[f"{variant}/{split}"] = dict(pool_equal=True, unique_ids=True,
                metadata_preserved_except_task_id=True, fake_assignments_unchanged=True,
                samples=len(flattened))
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i+1:]:
            for key in ["utt_id", "audio_path"]:
                assert not ({r[key] for r in source_pools[first]} & {r[key] for r in source_pools[second]})
    output.mkdir(parents=True)
    shutil.copytree(source / "master", output / "master")
    if (source / "generator_taxonomy.csv").exists():
        shutil.copy2(source / "generator_taxonomy.csv", output / "generator_taxonomy.csv")
    for (variant, split, i), (fields, rows, directory) in prepared.items():
        dest = output / variant / directory / f"{split}.csv"
        if VARIANTS[variant][1] == "unchanged":
            src = layouts[VARIANTS[variant][0]][i] / f"{split}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            assert digest(src) == digest(dest)
        else:
            write(dest, fields, rows)
    # Read back every emitted manifest independently.
    for variant, (layout, _) in VARIANTS.items():
        for split in SPLITS:
            loaded = [r for d in layouts[layout] for r in read(output / variant / d.name / f"{split}.csv")[1]]
            assert identity(loaded) == identity(source_pools[split])
    assert source_hashes == {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    out_hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob("*.csv")}
    report = dict(source=str(source), source_files_unchanged=True, seed=2026,
        split_pool_identity_checks=checks, split_utterance_and_path_disjoint=True,
        speaker_split_membership_unchanged=True, counts=count_rows,
        protocol3_allocation=allocation_rows, source_sha256=source_hashes, manifest_sha256=out_hashes)
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write(output / "protocol3_domain_allocation.csv", list(allocation_rows[0]), allocation_rows)
    table_rows = [{**r, "real_by_domain": json.dumps(r["real_by_domain"], sort_keys=True),
                   "fake_by_domain": json.dumps(r["fake_by_domain"], sort_keys=True)} for r in count_rows]
    write(output / "all_protocol_counts.csv", list(table_rows[0]), table_rows)
    lines = ["# Five protocols from one locked sample pool", "",
        "Only task assignment changes. Train/dev/eval membership, audio paths, labels, and",
        "speaker split membership are preserved. No audio is added, removed, or reused.",
        "Protocols 4 and 5 preserve the mixed-real source assignments.", "",
        "## Definitions", "",
        "- Protocol 1: dataset-wise real / dataset-wise fake.",
        "- Protocol 2: dataset-wise real / mechanism-wise fake.",
        "- Protocol 3: source-support-matched real / mechanism-wise fake.",
        "- Protocol 4: four-domain mixed real / dataset-wise fake.",
        "- Protocol 5 (RAMI): four-domain mixed real / mechanism-wise fake.", "",
        "Protocol 3 allocates each domain's fixed real budget across tasks proportionally to that",
        "domain's Fake counts, independently within each split. Largest-remainder rounding",
        "with ascending-task tie breaks preserves integer totals. Utterance assignment uses",
        "SHA256(seed, split, domain, utt_id), not model results. Zero Fake support receives",
        "zero Real from that domain. This matches dataset-source support, not exact per-task",
        "domain proportions, utterance pairs, language, speaker, or acoustic provenance.", "",
        "Protocol 3 formula: R[t,d] = apportion(R[d] * F[t,d] / sum_t F[t,d]).",
        "Active assignment is task_id and directory. Other inherited task/taxonomy fields",
        "remain source provenance and must not be used to infer reassigned real membership.",
        "Changing eval grouping changes macro EER; compare final Pool EER on the shared pool.",
        "Identical pools alone do not ensure identical optimization budgets or update counts.", "",
        "## Counts", "", "| Protocol | Split | Task | Real | Fake | Total |",
        "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['protocol'].split('_')[0]} | {r['split']} | {r['task']} | {r['real']} | {r['fake']} | {r['total']} |" for r in count_rows]
    (output / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("PASS: 15 protocol/split pool-equality checks; input files unchanged")
    print("OUTPUT", output)
    for r in count_rows:
        if r["protocol"].startswith("protocol_5"):
            print(json.dumps(r))
if __name__ == "__main__":
    main()
