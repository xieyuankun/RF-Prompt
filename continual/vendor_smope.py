import torch
from torch import nn
from torch.nn import functional as F
class OnePrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, num_heads=12):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks

        self.e_p_length = int(prompt_param[0])
        self.topk = int(prompt_param[1])
        self.mu_router = float(prompt_param[2])
        self.mu_router_old = float(prompt_param[3])
        self.eps = float(prompt_param[4])
        self.e_layers = [0, 1, 2, 3, 4, 5]

        self.num_heads = num_heads
        head_dim = self.key_d // self.num_heads
        self.head_dim = head_dim
        self.num_experts = self.e_p_length // 2

        # e prompt init
        for e in self.e_layers:
            for l in range(self.num_experts):
                for h in range(self.num_heads):
                    p_k = tensor_prompt(1, head_dim)
                    setattr(self, f"e_pk_{e}_{l}_{h}", p_k)
                    p_v = tensor_prompt(1, head_dim)
                    setattr(self, f"e_pv_{e}_{l}_{h}", p_v)

                    freq = 0
                    setattr(self, f"e_freq_{e}_{l}_{h}", freq)

        self.num_samples = 0
        self.used_frequently = [
            [[False for _ in range(self.num_experts)] for _ in range(self.num_heads)]
            for _ in self.e_layers
        ]
        self.router_criterion = nn.CrossEntropyLoss()

    def process_task_count(self):
        self.task_count += 1
        self.save_old_prompts()

        for e in self.e_layers:
            for h in range(self.num_heads):
                freq_head = torch.tensor(
                    [
                        getattr(self, f"e_freq_{e}_{l}_{h}")
                        for l in range(self.num_experts)
                    ]
                ).to(torch.float32)
                freq_head_mean = freq_head.sum() / (freq_head > 0).sum().clamp(
                    min=1.0
                )  # avoid division by zero
                for l in range(self.num_experts):
                    freq = getattr(self, f"e_freq_{e}_{l}_{h}")
                    if freq >= freq_head_mean:
                        self.used_frequently[e][h][l] = True

    def router_loss(self, prompt_scores, task_id=-1, topk=-1):
        loss = 0.0

        if self.mu_router > 0 and topk > 0:
            max_loss = 0
            for e in self.e_layers:
                prompt_score, prompt_score_label_ = prompt_scores[
                    e
                ]  # (B, num_heads, 1, num_prompt)
                _, indices = torch.topk(prompt_score_label_, self.topk, dim=-1)
                mask = torch.zeros_like(prompt_score).scatter(
                    -1, indices, 1.0
                )  # (B, num_heads, 1, num_prompt)
                not_mask = 1.0 - mask
                not_mask = not_mask * self.eps
                with torch.no_grad():
                    prompt_score_max = prompt_score.max(
                        dim=-1, keepdim=True
                    ).values  # (B, num_heads, 1, 1)
                    prompt_score_min = prompt_score.min(
                        dim=-1, keepdim=True
                    ).values  # (B, num_heads, 1, 1)
                prompt_score = prompt_score + not_mask * (
                    prompt_score_max - prompt_score_min
                )  # (B, num_heads, 1, num_prompt)

                for h in range(self.num_heads):
                    indices_h = indices[:, h, 0, :]  # (B, topk)
                    for i in range(self.topk):
                        max_loss += (
                            self.router_criterion(
                                prompt_score[:, h, 0, :], indices_h[:, i]
                            )
                            * self.mu_router
                        )

            loss += max_loss

        if task_id > 0 and self.mu_router_old > 0:
            for e in self.e_layers:
                for h in range(self.num_heads):
                    current_pk_h = []
                    old_pk_h = []
                    sampled = []

                    for l in range(self.num_experts):
                        current_pk = getattr(self, f"e_pk_{e}_{l}_{h}")
                        old_pk = getattr(self, f"old_e_pk_{e}_{l}_{h}")
                        current_pk_h.append(current_pk)
                        old_pk_h.append(old_pk)
                        sampled.append(self.used_frequently[e][h][l])

                    current_pk_h = torch.cat(current_pk_h, dim=0)
                    old_pk_h = torch.cat(old_pk_h, dim=0)
                    sampled = torch.tensor(
                        sampled, device=current_pk_h.device, dtype=torch.bool
                    )
                    sampled_pk_h = old_pk_h[sampled]  # (num_used, head_dim)

                    old_logits = sampled_pk_h @ old_pk_h.t()  # (num_used, num_experts)
                    current_logits = (
                        sampled_pk_h @ current_pk_h.t()
                    )  # (num_used, num_experts)
                    _, indices = torch.topk(
                        old_logits, self.topk, dim=-1
                    )  # (num_used, topk)

                    for i in range(self.topk):
                        loss += (
                            self.router_criterion(current_logits, indices[:, i])
                            * self.mu_router_old
                        )

        return loss

    def forward(self, x_querry, l, x_block, train=False, task_id=None, noise=False):
        e_valid = False
        loss = 0.0

        if l in self.e_layers:
            e_valid = True
            B = x_block.shape[0]
            pk = []  # (num_heads, num_prompt, head_dim)
            pv = []
            eps_decay = []
            for h in range(self.num_heads):
                pk_h = []
                pv_h = []
                eps_decay_h = []
                for i in range(self.num_experts):
                    _pk_h = getattr(self, f"e_pk_{l}_{i}_{h}")
                    _pv_h = getattr(self, f"e_pv_{l}_{i}_{h}")
                    pk_h.append(_pk_h)
                    pv_h.append(_pv_h)
                    freq = getattr(self, f"e_freq_{l}_{i}_{h}")
                    if train and self.used_frequently[l][h][i]:
                        eps_decay_h.append(self.eps)
                    else:
                        if not train and freq == 0:
                            eps_decay_h.append(2.0)
                        else:
                            eps_decay_h.append(0.0)

                pk_h = torch.cat(pk_h, dim=0).unsqueeze(0)  # (1, num_prompt, head_dim)
                pv_h = torch.cat(pv_h, dim=0).unsqueeze(0)  # (1, num_prompt, head_dim)

                pk.append(pk_h)  # (num_heads, num_prompt, head_dim)
                pv.append(pv_h)  # (num_heads, num_prompt, head_dim)
                eps_decay.append(eps_decay_h)

            pk = torch.cat(pk, dim=0)  # (num_heads, num_prompt, head_dim)
            pv = torch.cat(pv, dim=0)  # (num_heads, num_prompt, head_dim)
            eps_decay = torch.tensor(
                eps_decay, device=pk.device, dtype=torch.float32
            )  # (num_heads, num_experts)
            Ek = pk.unsqueeze(0).expand(B, -1, -1, -1)
            Ev = pv.unsqueeze(0).expand(B, -1, -1, -1)

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev, eps_decay]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

    def print_freq(self):
        print("-" * 20)
        print(f"Num Samples: {self.num_samples}")
        for e in self.e_layers:
            for h in range(self.num_heads):
                print("-" * 10)
                print(f"Layer {e} Head {h}:")
                freq = []
                for l in range(self.num_experts):
                    freq.append(getattr(self, f"e_freq_{e}_{l}_{h}"))
                print(freq)
        print("-" * 20)

    def update_num_samples(self, num_samples):
        self.num_samples += num_samples

    def update_prompt(self, prompt_scores):
        if self.topk > 0:
            for e in self.e_layers:
                prompt_score, _ = prompt_scores[e]  # (B, num_heads, 1, num_prompt)
                for h in range(self.num_heads):
                    weight = prompt_score[:, h, 0, :]  # (B, num_prompt)
                    _, indices = torch.topk(weight, self.topk, dim=-1)
                    indices = indices.reshape(-1)

                    unique_vals, counts = torch.unique(indices, return_counts=True)

                    for u, c in zip(unique_vals.tolist(), counts.tolist()):
                        freq = getattr(self, f"e_freq_{e}_{u}_{h}")
                        setattr(self, f"e_freq_{e}_{u}_{h}", freq + c)

    def save_old_prompts(self):
        print("Saving old prompts")
        # Save old prompts
        for e in self.e_layers:
            for l in range(self.num_experts):
                for h in range(self.num_heads):
                    pv_h = getattr(self, f"e_pv_{e}_{l}_{h}")
                    pv_h = pv_h.detach().clone()
                    setattr(self, f"old_e_pv_{e}_{l}_{h}", pv_h)

                    pk_h = getattr(self, f"e_pk_{e}_{l}_{h}")
                    pk_h = pk_h.detach().clone()
                    setattr(self, f"old_e_pk_{e}_{l}_{h}", pk_h)

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p
