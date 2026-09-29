"""Token-wise Real Prompt cosine anchoring (no feature teacher forward)."""
import torch.nn.functional as F


def prompt_cosine_anchor(current, reference, layer_count):
    """Mean over corresponding tokens and selected layers, not a summed loss."""
    if current.shape != reference.shape or current.ndim != 3:
        raise ValueError("Expected equal [layers, tokens, hidden] prompt tensors")
    if not 1 <= layer_count <= current.shape[0]:
        raise ValueError("Protected layer count must be within the prompt depth")
    return (1.0 - F.cosine_similarity(
        current[:layer_count], reference[:layer_count].detach(), dim=-1,
    )).mean()
