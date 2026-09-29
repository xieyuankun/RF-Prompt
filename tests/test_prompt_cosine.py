import inspect

import pytest
import torch

from continual.prompt_cosine import prompt_cosine_anchor
from continual.orthogonal_prompt import RFPromptEncoder


def test_cosine_anchor_only_updates_selected_current_layers():
    torch.manual_seed(7)
    current = torch.randn(4, 2, 8, requires_grad=True)
    reference = torch.randn(4, 2, 8, requires_grad=True)
    expected = (1 - torch.nn.functional.cosine_similarity(
        current[:2], reference[:2].detach(), dim=-1
    )).mean()
    loss = prompt_cosine_anchor(current, reference, 2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert reference.grad is None
    assert current.grad[:2].abs().sum() > 0
    assert torch.count_nonzero(current.grad[2:]) == 0


@pytest.mark.parametrize('depth', [0, 5])
def test_cosine_anchor_rejects_invalid_depth(depth):
    prompt = torch.randn(4, 2, 8)
    with pytest.raises(ValueError):
        prompt_cosine_anchor(prompt, prompt, depth)


def test_retired_switches_are_not_constructor_options():
    parameters = inspect.signature(RFPromptEncoder).parameters
    for name in (
        'router_v2', 'centroid_router', 'incremental_real',
        'real_domain_residual', 'real_sparse_pool_tokens',
        'gap_temperature_start', 'layer_top_k',
    ):
        assert name not in parameters


def test_retired_state_is_not_stored_on_encoder():
    source = inspect.getsource(RFPromptEncoder.__init__)
    for name in (
        'real_task_prompts', 'real_domain_prompts',
        'real_domain_centroids', 'real_domain_names', 'task_prototypes',
        'fake_keys', 'fake_gate_heads', 'router_projector', 'router_heads',
    ):
        assert name not in source
