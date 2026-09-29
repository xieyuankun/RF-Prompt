# Adapted prompt baselines

The Table-1 prompt competitors are matched audio adaptations rather than scores
copied from their original image or domain-adaptation papers. They use RAMI,
frozen XLS-R 300M, the same seeded AASIST backend, 50 epochs per task, and the
same development-checkpoint rule as RF-Prompt.

- `oisoprompt`: the INTERSPEECH 2024 shallow input-prompt configuration.
- `singleprompt`: one shared K/V prompt in every XLS-R attention layer.
- `kaprompt`: four task prompts with top-1 key retrieval and inheritance.
- `smope`: 25 experts per head, top five, in the first six layers.
- `rainbow`: an additional development baseline, not a main-table method.

SMoPE is derived from commit
`27b982c5d8ff41c345df44e3dd4035b2348548e0`; RainbowPrompt is derived from
commit `0b84d3e3738669fe4bb516ccc1c526a535576ad3`. See
`scripts/run_prompt_baseline_rami.sh` for the matched launch command.
