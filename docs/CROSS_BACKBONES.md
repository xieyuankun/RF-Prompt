# Cross-backbone runs

The paper compares RF-Prompt and Sequential under the same backbone family.
Use `scripts/run_backbone_rami.sh` with one of these family identifiers:

| Paper backbone | Family | Model directory |
|---|---|---|
| XLS-R 300M / 1B / 2B | `xlsr` | corresponding local Hugging Face directory |
| WavLM-Large | `wavlm` | `microsoft/wavlm-large` |
| W2V-BERT 2.0 | `w2vbert` | `facebook/w2v-bert-2.0` |

W2V-BERT 2.0 requires `transformers==4.46.3`, matching the environment used
for that experiment. The primary XLS-R release remains pinned to 4.36.2.

Example:

```bash
bash scripts/run_backbone_rami.sh rfprompt wavlm \
  /models/wavlm-large /data/protocol_5_rami outputs/wavlm_rfprompt
bash scripts/run_backbone_rami.sh sequential wavlm \
  /models/wavlm-large /data/protocol_5_rami outputs/wavlm_sequential
```

The prompt width and AASIST input projection are read from the backbone config.
The public configuration uses the first 24 layers for real-prompt consistency;
XLS-R 1B/2B still create prompt and residual parameters for every encoder layer.
All other optimization settings match the primary RAMI run.
