# Release cleanup

This release removes disabled experimental routes: prototype-replay routing,
centroid routing, layer-top-k/GAP fake routing, sparse real-prompt pools,
incremental real-prompt banks, and domain-specific real residuals. Their
constructor switches, CLI switches, call sites and dedicated tests were removed.

The audio loader, augmentations, SSL loader, AASIST implementation, baseline
implementations, optimizer, scheduler and checkpoint selection are preserved.
The prompt-cosine implementation and its training integration are restored from
the saved experiment code, rather than reimplemented.

Dead key, prototype, task-real, domain-real and gate-head attributes are not
stored in the released model. To preserve the exact seeded initialization used
by the experiments, `add_task()` consumes the same retired key and gate-head RNG
draws and immediately discards them. Consequently later Fake Prompt parameters
remain numerically identical without exposing unused checkpoint state. Older
experiment checkpoints still load with `strict=False`; retired keys are ignored.
The internal checkpoint identifier remains `oprompt`.

CPU regression with a small 24-layer Wav2Vec2 model and a lightweight test backend
compared the original experiment implementation with this release across four
tasks, two Adam updates per task. Response fusion with prompt cosine, uniform
fusion with prompt cosine, and response fusion with feature SPD matched exactly
for initialization, RNG state, trainable flags, logits, losses, gradients and
updated paper-model state. Loading historical checkpoints while ignoring only
retired compatibility keys was also checked.

These are implementation-equivalence tests, not a new full-dataset EER run.
Full reproduction still requires the documented data, pretrained weights and
training configuration.
