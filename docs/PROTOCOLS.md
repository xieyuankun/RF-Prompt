# Controlled continual-learning protocols

All five protocols use the same global training, development, and evaluation
sample pools. Only task assignment changes.

| Protocol | Real arrival | Fake organization |
|---|---|---|
| 1 | Dataset-wise | Dataset-wise |
| 2 | Dataset-wise | Mechanism-wise |
| 3 | Source-support-matched | Mechanism-wise |
| 4 | Four-domain mixture | Dataset-wise |
| 5 (RAMI) | Four-domain mixture | Mechanism-wise |

Mechanism-wise tasks follow the paper taxonomy: M1 classical pipeline, M2
neural acoustic pipeline, M3 neural codec pipeline, and M4 speech-LM pipeline.
Real utterances remain disjoint across tasks in the recurring-mixture settings.

The protocol builder expects this locked source layout:

```text
locked_source/
├── master/{train,dev,eval}.csv
├── protocol_a/a0_asv19/{train,dev,eval}.csv
├── protocol_a/a1_asv5/{train,dev,eval}.csv
├── protocol_a/a2_codecfake/{train,dev,eval}.csv
├── protocol_a/a3_atadd_t2_speech/{train,dev,eval}.csv
├── protocol_b/b0_classical_waveform/{train,dev,eval}.csv
├── protocol_b/b1_neural_vocoder/{train,dev,eval}.csv
├── protocol_b/b2_neural_codec/{train,dev,eval}.csv
└── protocol_b/b3_codec_token_alm/{train,dev,eval}.csv
```

The trainer only requires `utt_id`, `audio_path`, and binary `label` (`0` real,
`1` fake). The five-protocol builder additionally requires `dataset` and
`task_id`, because it reallocates real speech by source domain while retaining
the fake-task assignment. Its accepted `dataset` values are exactly
`asv19_la`, `asvspoof5_track1`, `codecfake`, and `atadd_track2_speech`.

The builder is not a raw-dataset parser. Before running it, create the locked
`master`, dataset-wise `protocol_a`, and mechanism-wise `protocol_b` manifests
shown above. Every split must contain identical global utterance membership in
all three views; utterance IDs and audio paths must be unique within a split and
disjoint across train, development, and evaluation. Additional provenance
columns are preserved.
