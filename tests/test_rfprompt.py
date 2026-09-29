import torch
import torch.nn as nn

from continual.orthogonal_prompt import (
    RFPromptEncoder, response_soft_weights,
)
from continual.trainer import RealProtectedPromptMethod


def make_prompt_state(
    layers=2, tokens=2, hidden=16,
):
    encoder = RFPromptEncoder.__new__(RFPromptEncoder)
    nn.Module.__init__(encoder)
    encoder.num_layers = layers
    encoder.hidden_size = hidden
    encoder.real_tokens = tokens
    encoder.fake_tokens = tokens
    encoder.router_temperature = 0.1
    encoder.router_floor = 0.1
    encoder.use_shared_real = True
    encoder.uniform_fake_router = False
    encoder.joint_mode = 'none'
    encoder.fake_init_mode = 'orthogonal'
    encoder.fake_init_scale = 0.1
    encoder.collect_layer_features = False
    encoder.router_query_size = hidden
    encoder.real_prompt = nn.Parameter(torch.randn(layers, tokens, hidden))
    encoder.group_router = nn.Linear(hidden, 2)
    encoder.fake_prompts = nn.ParameterList()
    encoder.fake_prompt_bases = nn.ParameterList()
    encoder.active_tasks = 0
    encoder.training_task = 0
    encoder.add_task()
    encoder.set_task(0)
    return encoder


def test_fake_bank_grows_only_when_new_task_arrives():
    encoder = make_prompt_state()
    assert encoder.task_count == 1
    encoder.set_task(2)
    assert encoder.task_count == 3
    assert [value.requires_grad for value in encoder.fake_prompts] == [
        False, False, True
    ]


def test_new_fake_prompt_is_initialized_in_old_orthogonal_complement():
    encoder = make_prompt_state()
    encoder.set_task(1)
    assert encoder.orthogonality_loss().item() < 1e-10


def test_orthogonal_loss_detects_overlap_and_detaches_old_prompts():
    encoder = make_prompt_state()
    encoder.set_task(1)
    with torch.no_grad():
        encoder.fake_prompts[1].copy_(encoder.fake_prompts[0])
    loss = encoder.orthogonality_loss()
    assert loss.item() > 0.1
    loss.backward()
    assert encoder.fake_prompts[0].grad is None
    assert encoder.fake_prompts[1].grad is not None


def test_router_reduces_task_axis_to_fixed_token_count():
    encoder = make_prompt_state()
    encoder.set_task(2)
    query = torch.randn(4, encoder.hidden_size)
    _, group_weights, _, beta = encoder._router(query)
    bank = torch.stack(list(encoder.fake_prompts))
    mixed = torch.einsum('bt,tlph->blph', beta, bank)
    assert beta.shape == (4, 3)
    assert group_weights.shape == (4, 2)
    assert mixed.shape == (4, 2, encoder.fake_tokens, encoder.hidden_size)


def test_response_soft_moe_reduces_bank_to_fixed_token_count():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bank = torch.zeros(2, 3, 5, 2)
    bank[0, :, :, 0] = 1.0
    bank[1, :, :, 1] = 1.0
    scores, weights = response_soft_weights(query, bank, temperature=0.1)
    mixed = torch.einsum('bt,tlph->blph', weights, bank)
    assert scores.shape == (2, 2)
    assert weights.argmax(dim=1).tolist() == [0, 1]
    assert mixed.shape == (2, 3, 5, 2)


def test_real_layer_masks_cover_fixed_and_gap_topk_modes():
    router = {
        'gap_gate_values': torch.tensor([[[0.1, 0.9, 0.8, 0.2]]])
    }
    method = RealProtectedPromptMethod(real_update_layers='gap_topk')
    method.task_index = 0
    masks = method._real_layer_masks(router, 1, 4, torch.device('cpu'))
    assert masks.sum().item() == 4
    method.real_update_top_k = 2
    masks = method._real_layer_masks(router, 1, 4, torch.device('cpu'))
    assert masks.tolist() == [[0.0, 1.0, 1.0, 0.0]]
    method.real_update_layers = 'late8'
    masks = method._real_layer_masks(router, 2, 4, torch.device('cpu'))
    assert torch.equal(masks, torch.ones(2, 4))


def test_inherited_fake_prompt_records_parent_and_residual():
    encoder = make_prompt_state(hidden=8)
    encoder.fake_init_mode = 'inherit_response'
    with torch.no_grad():
        encoder.fake_prompts[0].fill_(0.0)
        encoder.fake_prompts[0][..., 0] = 1.0
    encoder.set_task(1)
    selected, _ = encoder.initialize_fake_prompt(
        torch.tensor([1.0] + [0.0] * 7)
    )
    assert selected == 0
    assert torch.allclose(
        encoder.fake_prompt_bases[1], encoder.fake_prompts[0]
    )
    assert not torch.allclose(
        encoder.fake_prompts[1], encoder.fake_prompt_bases[1]
    )


def test_real_anchor_is_soft_and_gradient_projection_removes_old_direction():
    prompt = torch.tensor([2.0, 1.0], requires_grad=True)
    anchor = torch.tensor([1.0, 1.0])
    penalty = RealProtectedPromptMethod._relative_anchor_penalty(prompt, anchor)
    penalty.backward()
    assert prompt.grad[0] != 0
    assert prompt.grad[1] == 0

    class PromptHolder:
        pass

    class ModelHolder:
        pass

    holder = PromptHolder()
    holder.real_prompt = nn.Parameter(torch.zeros(2))
    holder.real_prompt.grad = torch.tensor([2.0, 3.0])
    model = ModelHolder()
    model.prompt_encoder = holder
    method = RealProtectedPromptMethod(protection='gradient_projection')
    method.real_gradient_basis = torch.tensor([[1.0, 0.0]])
    method.after_backward(model)
    assert torch.allclose(holder.real_prompt.grad, torch.tensor([0.0, 3.0]))


def test_layerwise_pcgrad_only_removes_conflicting_fake_component():
    class PromptHolder:
        pass

    class ModelHolder:
        pass

    holder = PromptHolder()
    holder.real_prompt = nn.Parameter(torch.zeros(2, 1, 2))
    holder.real_prompt.grad = torch.tensor([
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ])
    model = ModelHolder()
    model.prompt_encoder = holder
    method = RealProtectedPromptMethod(protection='pcgrad_layerwise')
    method.task_index = 1
    method.pcgrad_real = torch.tensor([
        [[1.0, 0.0]],
        [[1.0, 0.0]],
    ])
    method.pcgrad_fake = torch.tensor([
        [[-1.0, 1.0]],
        [[1.0, 1.0]],
    ])
    method.pcgrad_fake_fraction = 1.0
    method.after_backward(model)
    assert torch.allclose(
        holder.real_prompt.grad,
        torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]])
    )


def test_spd_gradient_is_applied_only_to_shared_real_prompt():
    class PromptHolder:
        pass

    class ModelHolder:
        pass

    holder = PromptHolder()
    holder.real_prompt = nn.Parameter(torch.zeros(2, 1, 2))
    holder.real_prompt.grad = torch.zeros_like(holder.real_prompt)
    model = ModelHolder()
    model.prompt_encoder = holder
    method = RealProtectedPromptMethod(
        protection='none', spd_weight=0.1
    )
    method.spd_gradient = torch.ones_like(holder.real_prompt)
    method.after_backward(model)
    assert torch.allclose(
        holder.real_prompt.grad,
        torch.full_like(holder.real_prompt, 0.1)
    )


if __name__ == '__main__':
    test_fake_bank_grows_only_when_new_task_arrives()
    test_new_fake_prompt_is_initialized_in_old_orthogonal_complement()
    test_orthogonal_loss_detects_overlap_and_detaches_old_prompts()
    test_router_reduces_task_axis_to_fixed_token_count()
    test_response_soft_moe_reduces_bank_to_fixed_token_count()
    test_real_anchor_is_soft_and_gradient_projection_removes_old_direction()
    test_v2_router_is_hard_and_rehearses_old_prototypes()
    test_centroid_router_freezes_keys_and_shared_real_after_t0()
    test_centroid_router_uses_task_during_training_and_keys_at_inference()
    test_learnable_centroid_only_updates_current_key()
    test_real_domain_residual_only_trains_new_domains()
    test_real_domain_hard_routing_uses_frozen_centroids()
    test_real_expert_selection_rules()
    print('13 prompt tests passed')
