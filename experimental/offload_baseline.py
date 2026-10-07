"""Stage 2: no-cache expert offloading.

Experts live in pinned CPU memory. Every time a layer needs an expert, it is
copied to the GPU, used once, and thrown away. Everything else (attention,
router, embeddings) stays on the GPU.

The script checks correctness by generating text with the normal model first,
then offloading, generating again, and confirming the tokens are identical.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PROMPT = "The capital of France is"
NEW_TOKENS = 20


class OffloadedExperts(nn.Module):
    """Drop-in replacement for transformers' OlmoeExperts.

    Same forward signature and same math, but the weights stay on the CPU and
    each expert is fetched to the GPU only when a token is routed to it.
    """

    def __init__(self, original):
        super().__init__()
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
        # Plain attributes rather than nn.Parameters, so that model.to("cuda")
        # or similar calls can never silently move them back to the GPU.
        # Shapes: gate_up (64, 2048, 2048), down (64, 2048, 1024).
        # Indexing [e] gives one contiguous block per expert.
        self.gate_up_cpu = original.gate_up_proj.detach().cpu().pin_memory()
        self.down_cpu = original.down_proj.detach().cpu().pin_memory()
        # Simple counters. Stage 4 replaces these with a proper harness.
        self.fetches = 0
        self.bytes_moved = 0

    def fetch(self, e):
        """Copy expert e's weights to the GPU.

        This is the one place weights cross PCIe. In stage 3 the cache will
        sit here: check the cache first, and only copy on a miss.
        """
        gate_up = self.gate_up_cpu[e].to("cuda", non_blocking=True)
        down = self.down_cpu[e].to("cuda", non_blocking=True)
        self.fetches += 1
        self.bytes_moved += gate_up.nbytes + down.nbytes
        return gate_up, down

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Mirrors OlmoeExperts.forward from transformers 5.19 line for line.
        # The only change: weights come from self.fetch(e) instead of
        # self.gate_up_proj[e] / self.down_proj[e].
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            expert_hit = expert_hit.flatten().tolist()  # plain ints, needed to index CPU tensors

        for e in expert_hit:
            if e == self.num_experts:
                continue
            gate_up_w, down_w = self.fetch(e)
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down_w)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def generate(model, tok):
    """Greedy generation, timed. Returns (token ids, seconds)."""
    inputs = tok(PROMPT, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # 1. Reference run: whole model on the GPU.
    generate(model, tok)  # warm-up, so the timed run doesn't include one-time CUDA setup
    ref_out, ref_time = generate(model, tok)
    print(f"[GPU-resident] {torch.cuda.memory_allocated()/2**30:.2f} GiB on GPU, "
          f"{NEW_TOKENS/ref_time:.1f} tok/s")
    print(f"  text: {tok.decode(ref_out[0])!r}")

    # 2. Offload: swap every layer's experts module for our CPU-backed one.
    for layer in model.model.layers:
        layer.mlp.experts = OffloadedExperts(layer.mlp.experts)
    torch.cuda.empty_cache()

    # 3. Offloaded run.
    off_out, off_time = generate(model, tok)
    print(f"[Offloaded]    {torch.cuda.memory_allocated()/2**30:.2f} GiB on GPU, "
          f"{NEW_TOKENS/off_time:.1f} tok/s")
    print(f"  text: {tok.decode(off_out[0])!r}")

    # 4. Correctness and transfer totals.
    print(f"\nIdentical output: {torch.equal(ref_out, off_out)}")
    experts = [layer.mlp.experts for layer in model.model.layers]
    fetches = sum(x.fetches for x in experts)
    gb = sum(x.bytes_moved for x in experts) / 1e9
    print(f"Expert fetches: {fetches}  |  moved over PCIe: {gb:.1f} GB "
          f"({gb/NEW_TOKENS:.2f} GB per generated token)")


if __name__ == "__main__":
    main()