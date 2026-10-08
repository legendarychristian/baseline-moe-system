"""Stage 3b: expert offloading with a fixed-size GPU expert cache.

Builds on offload_baseline.py. Each layer gets its own cache of `num_slots`
experts, stored in GPU buffers allocated once up front, so GPU memory for
experts is exactly 16 layers x num_slots x 12.6 MB.

num_slots = 0 reproduces the stage 2 no-cache baseline.
num_slots = 64 caches every expert, so after the first fill nothing moves.
"""
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from expert_cache import ExpertCache, LRUPolicy

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PROMPT = "The capital of France is"
NEW_TOKENS = 20
SLOT_COUNTS = [0, 8, 16, 32, 64]  # experts cached per layer, out of 64


class OffloadedExperts(nn.Module):
    """Drop-in replacement for OlmoeExperts, backed by CPU memory + GPU cache."""

    def __init__(self, original):
        super().__init__()
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
        self.gate_up_cpu = original.gate_up_proj.detach().cpu().pin_memory()
        self.down_cpu = original.down_proj.detach().cpu().pin_memory()
        self.expert_bytes = self.gate_up_cpu[0].nbytes + self.down_cpu[0].nbytes
        self.timer = None  # optional TransferTimer (measure.py), set by the harness
        self.set_cache(0)

    def set_cache(self, num_slots, policy_cls=LRUPolicy):
        """(Re)build the cache with num_slots GPU slots. Starts empty (cold)."""
        self.gate_up_slots = self.down_slots = None  # release old buffers
        self.hits = self.misses = self.bytes_moved = 0
        if num_slots == 0:
            self.cache = None
            return
        self.cache = ExpertCache(num_slots, policy_cls())
        # Allocated once. Loading an expert copies into a slot; it never
        # allocates, so memory use is fixed for the whole run.
        self.gate_up_slots = torch.empty((num_slots, *self.gate_up_cpu.shape[1:]),
                                         dtype=self.gate_up_cpu.dtype, device="cuda")
        self.down_slots = torch.empty((num_slots, *self.down_cpu.shape[1:]),
                                      dtype=self.down_cpu.dtype, device="cuda")

    def _timed(self):
        return self.timer.transfer() if self.timer else nullcontext()

    def preload_all(self):
        """Copy every expert into the cache, then zero the counters.

        Needs num_slots == num_experts. Gives the zero-transfer reference: the
        same code path as every offloaded run, with nothing left to fetch.
        """
        for e in range(self.num_experts):
            slot = self.cache.insert(e)
            self.gate_up_slots[slot].copy_(self.gate_up_cpu[e])
            self.down_slots[slot].copy_(self.down_cpu[e])
        self.hits = self.misses = self.bytes_moved = 0

    def fetch(self, e):
        """Return expert e's weights on the GPU, copying over PCIe only on a miss."""
        if self.cache is None:  # no-cache baseline, same as stage 2
            self.misses += 1
            self.bytes_moved += self.expert_bytes
            with self._timed():
                gate_up = self.gate_up_cpu[e].to("cuda", non_blocking=True)
                down = self.down_cpu[e].to("cuda", non_blocking=True)
            return gate_up, down

        slot = self.cache.lookup(e)
        if slot is None:
            slot = self.cache.insert(e)  # may evict another expert
            # Overwriting an evicted expert's slot is safe here because all
            # work runs on one CUDA stream, so this copy is queued after any
            # computation that read the old contents. Once prefetching moves
            # copies to a second stream, this needs explicit synchronization.
            with self._timed():
                self.gate_up_slots[slot].copy_(self.gate_up_cpu[e], non_blocking=True)
                self.down_slots[slot].copy_(self.down_cpu[e], non_blocking=True)
            self.misses += 1
            self.bytes_moved += self.expert_bytes
        else:
            self.hits += 1
        return self.gate_up_slots[slot], self.down_slots[slot]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Unchanged from stage 2: mirrors OlmoeExperts.forward (transformers 5.19).
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            expert_hit = expert_hit.flatten().tolist()

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

    generate(model, tok)  # warm-up
    ref_out, ref_time = generate(model, tok)
    print(f"{'config':>14} | {'GPU GiB':>7} | {'tok/s':>5} | {'hit rate':>8} | "
          f"{'GB moved':>8} | identical")
    print(f"{'GPU-resident':>14} | {torch.cuda.memory_allocated()/2**30:7.2f} | "
          f"{NEW_TOKENS/ref_time:5.1f} |        - |        - | -")

    experts = []
    for layer in model.model.layers:
        layer.mlp.experts = OffloadedExperts(layer.mlp.experts)
        experts.append(layer.mlp.experts)
    torch.cuda.empty_cache()

    for slots in SLOT_COUNTS:
        for x in experts:
            x.set_cache(slots)
        torch.cuda.empty_cache()

        out, t = generate(model, tok)
        hits = sum(x.hits for x in experts)
        misses = sum(x.misses for x in experts)
        gb = sum(x.bytes_moved for x in experts) / 1e9
        print(f"{f'{slots} slots/layer':>14} | {torch.cuda.memory_allocated()/2**30:7.2f} | "
              f"{NEW_TOKENS/t:5.1f} | {hits/(hits+misses):8.1%} | {gb:8.1f} | "
              f"{torch.equal(ref_out, out)}")


if __name__ == "__main__":
    main()