"""Stage 4a: the benchmark workload, with fixed reference tokens.

Problem this solves: in bf16, two candidate tokens are sometimes tied to within
rounding error, and tiny numerical differences between code paths (grouped_mm
vs. our eager loop, different attention kernels) can flip which one wins. After
a flip, two runs continue *different text*, route to different experts, and do
different amounts of work, so they can't be compared fairly.

The fix:
  1. Generate each prompt's continuation ONCE (free-running greedy) and save
     the tokens to reference_tokens.json. Commit that file.
  2. Every benchmark run *replays* those exact tokens: at each decode step the
     model is fed the reference token, not its own prediction. So every
     configuration processes identical text and does identical work.
  3. Correctness = how often the model's own prediction agrees with the
     reference token (should be ~100%).
"""
import json
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from offload import OffloadedExperts

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
WORKLOAD = "prompts.json"
REFERENCE = "reference_tokens.json"


def load_workload(path=WORKLOAD):
    with open(path) as f:
        return json.load(f)


def eos_ids(model):
    eos = model.generation_config.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos or [])


def pick(logits, blocked_ids):
    """Greedy choice of the next token, never choosing end-of-sequence."""
    logits = logits[:, -1, :].clone()
    logits[:, blocked_ids] = float("-inf")
    return logits.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def greedy_generate(model, input_ids, new_tokens, blocked_ids):
    """Free-running greedy decoding. Only used once, to create the reference."""
    out = model(input_ids=input_ids, use_cache=True)              # prefill
    past = out.past_key_values
    next_tok = pick(out.logits, blocked_ids)
    generated = [next_tok]
    for _ in range(new_tokens - 1):                               # decode
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = pick(out.logits, blocked_ids)
        generated.append(next_tok)
    return torch.cat(generated, dim=1)  # generated tokens only, shape (1, new_tokens)


@torch.no_grad()
def replay(model, input_ids, ref_generated, blocked_ids, after_prefill=None):
    """Teacher-forced decoding: feed the reference tokens, record the model's predictions.

    The only difference from greedy_generate: each decode step is fed
    ref_generated[i] instead of the model's own previous prediction.
    after_prefill, if given, is called between the two phases (used for timing).
    """
    out = model(input_ids=input_ids, use_cache=True)              # prefill
    past = out.past_key_values
    preds = [pick(out.logits, blocked_ids)]
    if after_prefill:
        after_prefill()
    for i in range(ref_generated.shape[1] - 1):                   # decode
        out = model(input_ids=ref_generated[:, i:i + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        preds.append(pick(out.logits, blocked_ids))
    return torch.cat(preds, dim=1)


def make_reference(model, tok, workload, blocked):
    reference = {
        "model_id": MODEL_ID,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "experts_implementation": getattr(model.config, "_experts_implementation_internal", None),
        "new_tokens": workload["new_tokens"],
        "prompts": {},
    }
    for p in workload["prompts"]:
        ids = tok(p["text"], return_tensors="pt")["input_ids"].cuda()
        gen = greedy_generate(model, ids, workload["new_tokens"], blocked)
        reference["prompts"][p["id"]] = {
            "category": p["category"],
            "prompt_ids": ids[0].tolist(),
            "generated_ids": gen[0].tolist(),
        }
    return reference


def load_reference(path=REFERENCE):
    with open(path) as f:
        return json.load(f)


def check_agreement(model, reference, blocked, label):
    print(f"\n[{label}]")
    agree = total = 0
    for pid, r in reference["prompts"].items():
        ids = torch.tensor([r["prompt_ids"]], device="cuda")
        gen = torch.tensor([r["generated_ids"]], device="cuda")
        preds = replay(model, ids, gen, blocked)
        same = (preds == gen).sum().item()
        agree += same
        total += gen.numel()
        print(f"{pid:>13}: {same:3d}/{gen.numel()} tokens agree")
    print(f"overall agreement: {agree / total:.2%}")


def main():
    workload = load_workload()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    blocked = eos_ids(model)

    if os.path.exists(REFERENCE):
        reference = load_reference()
        print(f"Loaded existing {REFERENCE}")
    else:
        reference = make_reference(model, tok, workload, blocked)
        with open(REFERENCE, "w") as f:
            json.dump(reference, f)
        print(f"Created {REFERENCE}")

    # Same code path that created the reference: must be exactly 128/128 everywhere.
    check_agreement(model, reference, blocked, "GPU-resident (grouped_mm)")

    # Different numerical path (our eager loop): expect ~100%, maybe a near-tie or two.
    for layer in model.model.layers:
        layer.mlp.experts = OffloadedExperts(layer.mlp.experts)
        layer.mlp.experts.set_cache(64)
    torch.cuda.empty_cache()
    check_agreement(model, reference, blocked, "Offloaded, 64 slots/layer (eager loop)")


if __name__ == "__main__":
    main()