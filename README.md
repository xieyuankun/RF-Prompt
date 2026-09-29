<div align="center">

# RF-Prompt

### Learning as Deepfakes Evolve

**RF-Prompt for Continual Audio Deepfake Detection**

<p>
  <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv paper">
  <img src="https://img.shields.io/badge/Task-Continual_ADD-0b7285?style=for-the-badge" alt="Continual audio deepfake detection">
  <img src="https://img.shields.io/badge/Protocol-RAMI-f08c00?style=for-the-badge" alt="RAMI protocol">
</p>

<p>s
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/Backbone-XLS--R_300M-2f9e44?style=flat-square" alt="XLS-R 300M">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/Code-Official-2f9e44?style=flat-square&logo=github&logoColor=white" alt="Official code">
</p>

[Overview](#-overview) · [Highlights](#-highlights) · [Results](#-main-results) · [Quick Start](#-quick-start) · [Protocols](#-protocols) · [Reproduction](#-reproduction)

</div>

---

<p align="center">
  <img src="assets/rfprompt_overview.png" width="96%" alt="RF-Prompt overview">
</p>

## 🔎 Overview

Audio deepfake detectors must learn newly emerging generation mechanisms while
retaining previously acquired real- and fake-speech knowledge. This repository
provides:

- **RAMI**, a real-anchored mechanism-incremental protocol that introduces new
  fake mechanisms against a recurring mixed-domain real-speech background.
- **RF-Prompt**, an asymmetric continual-learning method with a shared real
  prompt and an expanding bank of task-specific fake experts.
- **Five controlled protocols** constructed from identical global training,
  development, and evaluation pools.
- **Matched baselines and evaluation**, including task-wise EER, pooled EER,
  and average forgetting.

## ✨ Highlights

| | Design | Purpose |
|---|---|---|
| 🧭 | **RAMI protocol** | Separates real-source arrival from fake-mechanism organization under a fixed sample pool. |
| 🛡️ | **Shared real prompt** | Preserves reusable real-speech knowledge through parameter-level cosine anchoring. |
| 🧩 | **Inherited fake experts** | Transfers a selected historical expert and learns a complementary residual for each new mechanism. |
| 🔀 | **Input-adaptive Soft MoE** | Fuses all complete fake experts into a fixed number of injected tokens without task identity. |
| 🔁 | **Replay-free updates** | Does not store historical training audio or require a feature-teacher forward pass. |

## 🏆 Main results

Final performance on RAMI with XLS-R 300M. All methods use the same task order,
sample pools, training budget, and evaluation procedure. Lower is better.

| Method | Avg EER ↓ | Pooled EER ↓ | AF ↓ |
|---|---:|---:|---:|
| Sequential | 13.130 | 13.445 | 6.433 |
| EWC | 12.255 | 12.805 | 5.827 |
| SinglePrompt | 11.665 | 12.105 | 5.593 |
| **RF-Prompt** | **10.110** | **10.370** | **4.560** |

> Values are percentages. Every task checkpoint is selected using development EER.

## 🚀 Quick Start

### 1. Create the environment

```bash
conda create -n rfprompt python=3.10 -y
conda activate rfprompt
pip install -r requirements.txt
pip install pytest
```

### 2. Download the SSL backbone

```bash
huggingface-cli download facebook/wav2vec2-xls-r-300m \
  --local-dir /path/to/wav2vec2-xls-r-300m
```

### 3. Run RF-Prompt on RAMI

```bash
bash scripts/run_rami.sh \
  /path/to/wav2vec2-xls-r-300m \
  /path/to/protocol_5_rami \
  /path/to/output
```

<details>
<summary><b>Primary configuration</b></summary>

- five shared real-prompt tokens per transformer layer;
- five adaptively fused fake-prompt tokens per transformer layer;
- 50 epochs per task and seed 2026;
- real-prompt cosine weight `1.0` over all 24 layers;
- fake-residual orthogonality weight `0.1` over all 24 layers;
- frozen XLS-R backbone with a trainable AASIST backend;
- best-development checkpoint propagation between tasks.

</details>

## 🗂️ Data manifests

Audio is not redistributed. Download each dataset from its official source and
follow its original license and access conditions.

### Dataset access

| Dataset | Subset used in this work | Official source |
|---|---|---|
| ASVspoof 2019 | Logical Access (LA) | [Zenodo](https://zenodo.org/records/6906306) (download and extract `LA.zip`) |
| ASVspoof 5 | Track 1 | [Zenodo](https://zenodo.org/records/14498691) |
| CodecFake | Training, development, and evaluation subsets | [Official repository and download table](https://github.com/xieyuankun/Codecfake) |
| AT-ADD | Track 2 speech | [Hugging Face](https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2) ([challenge instructions](https://www.at-add.com/instructions); access approval is required) |

Keep the extracted audio outside this Git repository. The following sibling
layout is recommended, but the directory names are not hard-coded:

```text
workspace/
├── RF-Prompt/                         # this repository
├── datasets/                          # raw/extracted audio
│   ├── ASVspoof2019_LA/
│   ├── ASVspoof5/
│   ├── CodecFake/
│   └── AT-ADD-Track2/
└── protocols/
    ├── five_protocols/                # extracted fixed Protocols 1--5
    └── locked_source/                 # optional inputs for rebuilding them
```

The paper's complete, fixed protocol manifests are provided in
[`protocols/rfprompt_protocols_seed2026.zip`](protocols/rfprompt_protocols_seed2026.zip).
Extract them once:

```bash
mkdir -p ../protocols
unzip protocols/rfprompt_protocols_seed2026.zip -d ../protocols
export RFPROMPT_DATA_ROOT=/path/to/workspace/datasets
```

The archive contains all five protocols, every train/development/evaluation
split, the generator taxonomy, sample counts, and SHA256 audit metadata. Its
SHA256 checksum is recorded in [`protocols/SHA256SUMS`](protocols/SHA256SUMS).

Paths inside the released manifests are portable and begin with one of the
four recommended dataset directory names shown above. When
`RFPROMPT_DATA_ROOT` is set, the loader resolves these paths from that root. An
absolute `audio_path` may still point anywhere on the local machine or server;
without the environment variable, a relative path is resolved from the CSV
directory.

### Training manifests

Each task split used directly by the trainer is represented by a CSV with at
least these three columns:

```csv
utt_id,audio_path,label
example_real,ASVspoof2019_LA/example_real.flac,0
example_fake,ASVspoof2019_LA/example_fake.flac,1
```

Here, `0` denotes real speech and `1` denotes fake speech. The RAMI training
root has the following layout:

```text
protocol_5_rami/
├── b0_classical_waveform/{train,dev,eval}.csv
├── b1_neural_vocoder/{train,dev,eval}.csv
├── b2_neural_codec/{train,dev,eval}.csv
└── b3_codec_token_alm/{train,dev,eval}.csv
```

### Rebuilding all five protocols (optional)

The released archive is sufficient for reproducing the experiments. To audit
or reconstruct the task assignment itself, prepare the locked source manifests
described in [docs/PROTOCOLS.md](docs/PROTOCOLS.md). These builder inputs
additionally contain `dataset` and `task_id`; accepted dataset names are
`asv19_la`, `asvspoof5_track1`, `codecfake`, and `atadd_track2_speech`. Then
run:

```bash
python scripts/build_five_protocols.py \
  --source ../protocols/locked_source \
  --output ../protocols/five_protocols
```

The generated RAMI directory is
`../protocols/five_protocols/protocol_5_rami` and can be passed directly as the
second argument of `scripts/run_rami.sh`.

## 🧭 Protocols

All protocols contain the same global samples; only task assignment changes.

| Protocol | real arrival | fake organization |
|---|---|---|
| 1 | Dataset-wise | Dataset-wise |
| 2 | Dataset-wise | Mechanism-wise |
| 3 | Source-support-matched | Mechanism-wise |
| 4 | Four-domain mixture | Dataset-wise |
| **5 (RAMI)** | **Four-domain mixture** | **Mechanism-wise** |

Build all five protocols from a locked source pool:

```bash
python scripts/build_five_protocols.py \
  --source /path/to/locked_source \
  --output /path/to/five_protocols
```

The builder writes sample counts, hashes, and pool-equality checks to
`audit.json`. See [docs/PROTOCOLS.md](docs/PROTOCOLS.md) for the expected source
layout and taxonomy.

## ♻️ Reproduction

### Matched continual-learning baselines

The training entry point exposes the following matched baselines. Citations
identify the original methods; every score in the paper is produced by our
audio/RAMI implementation under the matched training setup.

| CLI name | Original method | Where it is used in this project |
|---|---|---|
| `sequential` | Naive sequential fine-tuning | Main RAMI comparison, retention trajectories, limited-fake-data study, and cross-backbone study. |
| `ewc` | [Elastic Weight Consolidation (EWC)](https://doi.org/10.1073/pnas.1611835114) | Main RAMI comparison, retention trajectories, and limited-fake-data study. |
| `lwf` | [Learning without Forgetting (LwF)](https://doi.org/10.1007/978-3-319-46493-0_37) | Additional matched distillation reference documented in the appendix and released for reproduction; not a main-table result. |
| `owm` | [Orthogonal Weight Modification (OWM)](https://doi.org/10.1038/s42256-019-0080-x) | Main RAMI comparison and retention trajectories. |
| `rawm` | [RAWM](https://proceedings.mlr.press/v202/zhang23au.html) | ADD-specific baseline in the main RAMI comparison and retention trajectories. |
| `rwm` | [Radian Weight Modification (RWM)](https://ojs.aaai.org/index.php/AAAI/article/view/29929) | ADD-specific baseline in the main RAMI comparison, retention trajectories, and limited-fake-data study. |
| `rego` | [Region-Based Optimization (RegO)](https://ojs.aaai.org/index.php/AAAI/article/view/34535) | ADD-specific baseline in the main RAMI comparison, retention trajectories, and limited-fake-data study. |

Use the matched 50-epoch, batch-32 schedule:

```bash
bash scripts/run_baseline_rami.sh ewc \
  /path/to/wav2vec2-xls-r-300m \
  /path/to/protocol_5_rami \
  /path/to/output
```

The entry-point defaults match the primary paper configuration: RAMI layout,
50 epochs, batch size 32, development evaluation every epoch, response-based
fusion, inherited fake experts, real-prompt cosine weight 1.0, and residual
orthogonality weight 0.1.

### Adapted prompt baselines

The following methods were originally proposed for different continual-learning
or adaptation settings. We adapt them to the same frozen speech backbone,
binary ADD objective, RAMI task stream, and training budget.

| CLI name | Original method | Where it is used in this project |
|---|---|---|
| `oisoprompt` | [Oiso et al. prompt tuning](https://doi.org/10.21437/Interspeech.2024-81) | Audio-domain-adaptation reference in the main RAMI comparison and retention trajectories. |
| `singleprompt` | [SinglePrompt](https://openaccess.thecvf.com/content/CVPR2026F/html/Park_Is_Prompt_Selection_Necessary_for_Task-Free_Online_Continual_Learning_CVPRF_2026_paper.html) | Prompt-based CL baseline in the main RAMI comparison and retention trajectories. |
| `kaprompt` | [KA-Prompt](https://proceedings.mlr.press/v267/xu25as.html) | Prompt-based CL baseline in the main RAMI comparison and retention trajectories. |
| `smope` | [SMoPE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/099ba28b20462dac7bf057d21c27a011-Abstract-Conference.html) | Prompt-based CL baseline in the main RAMI comparison and retention trajectories. |
| `rainbow` | [RainbowPrompt](https://openaccess.thecvf.com/content/ICCV2025/html/Hong_RainbowPrompt_Diversity-Enhanced_Prompt-Evolving_for_Continual_Learning_ICCV_2025_paper.html) | Additional development implementation; released for inspection but not reported as a main-table result. |

Complete BibTeX entries for all cited baselines are provided in
[`CITATIONS.bib`](CITATIONS.bib). Implementation-specific adaptation details
are documented separately so that these results are not confused with scores
reported by the original papers.

```bash
bash scripts/run_prompt_baseline_rami.sh singleprompt \
  /path/to/wav2vec2-xls-r-300m \
  /path/to/protocol_5_rami \
  /path/to/output
```

Supported names are `oisoprompt`, `singleprompt`, `kaprompt`, `smope`, and
`rainbow`. See [adaptation details](docs/ADAPTED_PROMPT_BASELINES.md).

### Common-group protocol comparison

Every evaluation writes utterance-level scores. Regroup the final scores from
any protocol into the four RAMI groups used for the paper's common average:

```bash
python scripts/evaluate_common_groups.py \
  --scores /path/to/output/eval_scores_after_B3.csv \
  --rami_root /path/to/protocol_5_rami \
  --output /path/to/output/common_rami_metrics.json
```

The script reuses the same scores and verifies that RAMI exactly partitions the
evaluated pool.

### Alternative SSL backbones

RF-Prompt and Sequential support XLS-R 300M/1B/2B, WavLM-Large, and W2V-BERT
2.0. See [cross-backbone commands](docs/CROSS_BACKBONES.md).

### Outputs

```text
output/
├── best_dev_B0.pt ... best_dev_B3.pt
├── results.json
├── summary.json
├── eval_scores_after_B0.csv ... eval_scores_after_B3.csv
└── pooled_eval.csv
```

`summary.json` contains the final average EER, pooled EER, and average
forgetting. `results.json` retains the complete acquired-task trajectory.

### Tests

```bash
pytest -q tests
```

The release includes regression tests for expert growth, inheritance,
residual orthogonality, fixed-length soft fusion, real-prompt protection, and
routing behavior.

See [release cleanup notes](docs/CODE_CLEANUP.md) for removed experimental
switches, checkpoint compatibility and the scope of equivalence checks.

## 🧱 Code structure

```text
continual/orthogonal_prompt.py  # RF-Prompt encoder and expert bank
continual/trainer.py            # continual optimization and evaluation
CITATIONS.bib                   # source papers for released baselines
scripts/train_rfprompt.py       # RF-Prompt and matched baseline CLI
scripts/run_rami.sh             # exact primary configuration
scripts/run_baseline_rami.sh    # matched general-CL baselines
scripts/run_prompt_baseline_rami.sh # adapted prompt baselines
scripts/run_backbone_rami.sh    # matched cross-backbone runs
scripts/evaluate_common_groups.py # Table-2 common grouping
scripts/build_five_protocols.py # controlled protocol construction
dataset.py                      # portable manifest loader
tests/test_rfprompt.py          # regression tests
```

## 📄 Citation

If you use any code, protocol definition, or implementation from this
repository, please cite our paper:

```bibtex
@article{xie2026rfprompt,
  title   = {Learning as Deepfakes Evolve: RF-Prompt for Continual Audio
             Deepfake Detection},
  author  = {Xie, Yuankun and others},
  journal = {arXiv preprint},
  year    = {2026}
}
```

If you use a baseline implementation released in this repository, please also
cite the corresponding original work:

- General continual learning: EWC, LwF, and OWM.
- Continual audio deepfake detection: RAWM, RWM, and RegO.
- Prompt learning or audio adaptation: Oiso Prompt, SinglePrompt, KA-Prompt,
  SMoPE, and RainbowPrompt.

Copy-ready BibTeX entries for all of these methods are provided in
[`CITATIONS.bib`](CITATIONS.bib). The tables in the
[reproduction section](#️-reproduction) state where each implementation is
used in our experiments. The RF-Prompt entry above is provisional and should be
updated with the complete author list and arXiv identifier when available.

## 📜 License

Released under the [MIT License](LICENSE). Restricted datasets remain subject
to their original licenses and are not included in this repository.
