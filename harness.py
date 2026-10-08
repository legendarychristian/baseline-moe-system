"""Stage 4: run the fixed workload under each configuration, time it, save to CSV.

For every configuration, every prompt in reference_tokens.json is replayed
(same tokens every time). Each prompt starts with an empty cache, so prompts
are independent and the order of prompts doesn't matter.

For prefill and decode separately, we record:
  - wall-clock time
  - GPU time spent copying experts (CUDA events)  -> "waiting on transfers"
  - wall-clock minus transfer time                -> "everything else"
  - hits, misses, bytes moved

Results are printed and saved to results/run_<timestamp>.csv.
"""
import csv
import os
import time
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForCausalLM

from bench import MODEL_ID, eos_ids, load_reference, replay
from measure import TransferTimer, now
from offload import OffloadedExperts

SLOT_COUNTS = [0, 8, 16, 32, 64]


@dataclass
class Phase:
    seconds: float = 0.0
    transfer_ms: float = 0.0
    tokens: int = 0  # forward passes for decode; prompt tokens for prefill
    hits: int = 0
    misses: int = 0
    bytes: int = 0

    def transfer_share(self):
        return self.transfer_ms / 1000 / self.seconds if self.seconds else 0.0

    def hit_rate(self):
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


def counters(experts):
    return (sum(x.hits for x in experts), sum(x.misses for x in experts),
            sum(x.bytes_moved for x in experts))


def add(phase, seconds, transfer_ms, tokens, c_start, c_end):
    phase.seconds += seconds
    phase.transfer_ms += transfer_ms
    phase.tokens += tokens
    phase.hits += c_end[0] - c_start[0]
    phase.misses += c_end[1] - c_start[1]
    phase.bytes += c_end[2] - c_start[2]


def run_config(model, experts, reference, blocked, timer, setup):
    """Replay the whole workload once. setup() prepares the caches before each prompt."""
    prefill, decode = Phase(), Phase()
    agree = total = 0

    for i, r in enumerate(reference["prompts"].values()):
        ids = torch.tensor([r["prompt_ids"]], device="cuda")
        gen = torch.tensor([r["generated_ids"]], device="cuda")
        setup()                       # fresh cache for this prompt (not timed)
        if i == 0:
            torch.cuda.reset_peak_memory_stats()  # after the old config's buffers are freed
        timer.collect_ms()            # discard anything recorded during setup

        marks = {}

        def after_prefill():
            marks["t_prefill_end"] = now()
            marks["c_mid"] = counters(experts)
            marks["x_prefill"] = timer.collect_ms()
            marks["t_decode_start"] = now()  # excludes the bookkeeping above

        c0 = counters(experts)
        t0 = now()
        preds = replay(model, ids, gen, blocked, after_prefill=after_prefill)
        t_end = now()
        c_end = counters(experts)
        x_decode = timer.collect_ms()

        add(prefill, marks["t_prefill_end"] - t0, marks["x_prefill"], ids.shape[1], c0, marks["c_mid"])
        add(decode, t_end - marks["t_decode_start"], x_decode, gen.shape[1] - 1, marks["c_mid"], c_end)
        agree += (preds == gen).sum().item()
        total += gen.numel()

    return prefill, decode, agree / total, torch.cuda.max_memory_allocated() / 2**30


def print_header():
    print(f"{'config':>16} | {'GPU GiB':>7} | {'prefill s':>9} | {'pf xfer':>7} | "
          f"{'decode tok/s':>12} | {'dec hit':>7} | {'dec xfer':>8} | {'GB moved':>8} | "
          f"{'GB/s':>5} | agree")


def print_row(name, prefill, decode, agreement, gib):
    moved = prefill.bytes + decode.bytes
    xfer_s = (prefill.transfer_ms + decode.transfer_ms) / 1000
    gbps = f"{moved / 1e9 / xfer_s:5.1f}" if xfer_s else "    -"
    print(f"{name:>16} | {gib:7.2f} | {prefill.seconds:9.2f} | {prefill.transfer_share():7.0%} | "
          f"{decode.tokens / decode.seconds:12.1f} | {decode.hit_rate():7.1%} | "
          f"{decode.transfer_share():8.0%} | {moved / 1e9:8.1f} | {gbps} | {agreement:.2%}")


def csv_row(name, prefill, decode, agreement, gib):
    """Raw numbers only; percentages and tok/s can be computed from these later."""
    return {
        "config": name,
        "gpu_peak_gib": gib,
        **{f"prefill_{k}": v for k, v in asdict(prefill).items()},
        **{f"decode_{k}": v for k, v in asdict(decode).items()},
        "agreement": agreement,
    }


def main():
    reference = load_reference()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    blocked = eos_ids(model)
    timer = TransferTimer()

    # Warm-up: one prompt, so one-time CUDA setup isn't counted anywhere.
    first = next(iter(reference["prompts"].values()))
    replay(model, torch.tensor([first["prompt_ids"]], device="cuda"),
           torch.tensor([first["generated_ids"]], device="cuda"), blocked)

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/run_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = None

    def run(name, experts, setup):
        nonlocal writer
        result = run_config(model, experts, reference, blocked, timer, setup)
        print_row(name, *result)
        row = csv_row(name, *result)
        if writer is None:
            writer = csv.DictWriter(csv_file, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()  # save each row immediately, so a crash loses nothing

    print_header()

    # Reference 1: the original model, fused grouped_mm kernel, nothing offloaded.
    run("GPU-resident", [], lambda: None)

    experts = []
    for layer in model.model.layers:
        layer.mlp.experts = OffloadedExperts(layer.mlp.experts)
        layer.mlp.experts.timer = timer
        experts.append(layer.mlp.experts)
    torch.cuda.empty_cache()

    # Reference 2: our offloading code with every expert already on the GPU.
    def preloaded():
        for x in experts:
            x.set_cache(64)
            x.preload_all()
    run("ours, preloaded", experts, preloaded)

    # The actual sweep: cold cache at the start of every prompt.
    for slots in SLOT_COUNTS:
        def cold(slots=slots):
            for x in experts:
                x.set_cache(slots)
        run(f"{slots} slots/layer", experts, cold)

    csv_file.close()
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()