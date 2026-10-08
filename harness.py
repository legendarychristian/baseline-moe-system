"""Stage 4: run the fixed workload under each configuration, time it, save to CSV.

For every configuration, every prompt in reference_tokens.json is replayed
(same tokens every time). Each prompt starts with an empty cache, so prompts
are independent and the order of prompts doesn't matter.

For prefill and decode separately, we record:
  - wall-clock time
  - GPU time spent copying experts (CUDA events)  -> "waiting on transfers"
  - wall-clock minus transfer time                -> "everything else"
  - hits, misses, bytes moved

Results are printed and saved to two files:
  results/run_<timestamp>.csv      the 8 columns the report needs
  results/run_<timestamp>_raw.csv  every raw measurement, for recomputing anything else
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


def summary_row(name, prefill, decode, agreement, gib):
    """The 8 numbers the report needs."""
    return {
        "config": name,
        "gpu_peak_gib": round(gib, 2),
        "decode_tok_per_s": round(decode.tokens / decode.seconds, 2),
        "decode_hit_rate_pct": round(100 * decode.hit_rate(), 2) if decode.hits + decode.misses else "",
        "decode_gb_moved": round(decode.bytes / 1e9, 2),
        "decode_transfer_share_pct": round(100 * decode.transfer_share(), 2),
        "prefill_transfer_share_pct": round(100 * prefill.transfer_share(), 2),
        "agreement": round(agreement, 4),
    }


def raw_row(name, prefill, decode, agreement, gib):
    """Every raw measurement, so any other number can be recomputed later."""
    return {
        "config": name,
        "gpu_peak_gib": gib,
        **{f"prefill_{k}": v for k, v in asdict(prefill).items()},
        **{f"decode_{k}": v for k, v in asdict(decode).items()},
        "agreement": agreement,
    }


class CsvWriter:
    """Writes one row at a time, saving each immediately so a crash loses nothing."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()


def print_header():
    print(f"{'config':>16} | {'GPU GiB':>7} | {'decode tok/s':>12} | {'dec hit':>7} | "
          f"{'dec GB':>7} | {'dec xfer':>8} | {'pf xfer':>7} | agree")


def print_row(r):
    hit = f"{r['decode_hit_rate_pct']:6.1f}%" if r["decode_hit_rate_pct"] != "" else "      -"
    print(f"{r['config']:>16} | {r['gpu_peak_gib']:7.2f} | {r['decode_tok_per_s']:12.1f} | {hit} | "
          f"{r['decode_gb_moved']:7.1f} | {r['decode_transfer_share_pct']:7.1f}% | "
          f"{r['prefill_transfer_share_pct']:6.1f}% | {r['agreement']:.2%}")


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
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary = CsvWriter(f"results/run_{stamp}.csv")      # the 8 report columns
    raw = CsvWriter(f"results/run_{stamp}_raw.csv")      # every raw measurement

    def run(name, experts, setup):
        result = run_config(model, experts, reference, blocked, timer, setup)
        row = summary_row(name, *result)
        summary.write(row)
        raw.write(raw_row(name, *result))
        print_row(row)

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

    print(f"\nSaved {summary.path} and {raw.path}")


if __name__ == "__main__":
    main()