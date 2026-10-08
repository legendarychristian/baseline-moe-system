# MoE Expert Offloading Baseline

This project runs a Mixture-of-Experts model (OLMoE-1B-7B) with its experts stored in CPU RAM instead of GPU memory, and measures how much that slows it down.

---

## Part 1: The Caching System

### The model

OLMoE has 16 layers. Each layer has 64 experts, and a **router** picks 8 of them for every token. The experts make up about 93% of the model (around 12 GiB), but each token only uses a small fraction of them. Normally, the entire model sits on the GPU.

### What we changed

Inside each layer, the Hugging Face model has two separate parts:

1. **The router**, which decides which experts a token needs.
2. **The experts module**, which runs those experts.

We left the router alone and replaced only the experts module with our own version, `OffloadedExperts`. It receives the same inputs and does the same math. The only difference is **where the expert weights come from**:

- All expert weights live in **CPU RAM** (in pinned memory, which makes copies to the GPU faster).
- Everything else (attention, router, embeddings) stays on the **GPU**, which needs only about 0.9 GiB.
- When a token needs an expert, the code calls `fetch(e)` to get it onto the GPU.

### The cache

Each layer has a fixed number of **slots** on the GPU for holding experts. When `fetch(e)` is called:

- **Hit:** the expert is already in a slot, so it's used directly.
- **Miss:** the expert is copied from CPU RAM to the GPU over PCIe. If every slot is full, the **least recently used** expert is evicted to make room (LRU).

The slots are allocated once at the start, so GPU memory for experts is fixed: 16 layers × number of slots × 12.6 MB per expert.

- **0 slots** means no cache: every expert is copied every time it's needed.
- **64 slots** means every expert fits, so each one is copied at most once.

### Built for trying new policies

The cache is split into two pieces:

- **`ExpertCache`** manages the slots and counts hits and misses.
- **The policy** (currently `LRUPolicy`) only decides which expert to evict.

To try a new policy (LFU, MRS, and so on), you write a new policy class. Nothing else changes.

### Files

| File | What it does |
|---|---|
| `expert_cache.py` | The cache and the LRU policy |
| `offload.py` | `OffloadedExperts`, which replaces the model's experts module |

---

## Part 2: Benchmarking

### The workload

`prompts.json` contains 8 varied prompts (facts, code, math, a story, a dialogue, a long passage, a list, and Spanish). Each one generates 128 tokens. Variety matters because different topics use different experts.

### Fixed tokens

Small rounding differences can make two runs pick different words, which would make them process different text and do different amounts of work. To keep comparisons fair:

1. `bench.py` generates each prompt's text **once** and saves the tokens to `reference_tokens.json`.
2. Every benchmark run feeds the model **exactly those tokens**, so every configuration does identical work.
3. Correctness is checked by how often the model's own predictions agree with the saved tokens (the **agreement** column).

### What gets compared

| Configuration | What it is |
|---|---|
| GPU-resident | The original model, fully on the GPU. The best case. |
| ours, preloaded | Our code with every expert already on the GPU. Same code as the offloaded runs, but no copying. |
| 0 to 64 slots/layer | Offloaded runs with different cache sizes. |

Each prompt starts with an empty cache.

### What gets measured

Each run is split into **prefill** (processing the prompt) and **decode** (generating one token at a time), because they behave differently. For each, we record:

- **Total time**
- **Transfer time:** how long the GPU spent copying experts, measured with CUDA events (timestamps recorded by the GPU itself)
- **Hits, misses, and bytes moved**

From these, the main numbers are:

- **Decode tok/s:** how fast tokens are generated
- **Hit rate:** how often the needed expert was already cached
- **Transfer share:** the fraction of time spent waiting on copies
- **GPU memory:** peak memory used on the GPU

### How to run

```bash
python3 bench.py      # creates reference_tokens.json (only needed once)
python3 harness.py    # runs all configurations
```

The harness prints a table and saves the raw numbers to `results/run_<timestamp>.csv`.

### Files

| File | What it does |
|---|---|
| `prompts.json` | The 8 benchmark prompts |
| `bench.py` | Creates the saved tokens and replays them |
| `measure.py` | Timing helpers |
| `harness.py` | Runs every configuration and saves results |

---

## Part 3: Results Against the Project Objective

**Objective:** get a real MoE model running with expert offloading, and build the instrumentation needed to evaluate any future policy.

All numbers below come from one run on an NVIDIA A10 (24 GB, PCIe Gen4): 8 prompts × 128 tokens. Raw data: `results/run_20261007_031307.csv`.

### Deliverable 1: A working baseline ✅

- **Model:** OLMoE-1B-7B (64 experts per layer, 8 used per token, 16 layers).
- **Offloading:** all expert weights in CPU RAM, copied to the GPU on demand.
- **Policies:** no cache (0 slots) and LRU (8, 16, 32, and 64 slots per layer).
- **It's correct:** every offloaded configuration agrees with the original model on 99.02% of tokens, and on exactly the same tokens in every run. The remaining 10 out of 1,024 are near-ties where rounding decides the winner.

We wrote our own offloading layer in PyTorch instead of using Mixtral-offloading or llama.cpp, because every copy over PCIe then goes through our own code, which makes it easy to measure and to plug in new policies.

### Deliverable 2: A measurement harness ✅ (with two small gaps)

| Required metric | How it's measured | Status |
|---|---|---|
| Tokens/sec | Wall-clock time, prefill and decode measured separately | ✅ |
| Cache hit rate | Hits and misses counted inside `fetch()` | ✅ |
| Bytes moved between tiers | Every miss adds one expert (12.6 MB) to a counter | ✅ |
| Memory, GPU tier | Peak GPU memory per configuration | ✅ |
| Memory, CPU tier | Not measured yet. All experts always sit in CPU RAM: 16 × 64 × 12.6 MB ≈ **12 GiB**, the same for every configuration | ⚠️ calculated, not measured |
| Steady-state memory | Not recorded separately. Cache slots are allocated once at the start, so steady-state should be very close to peak | ⚠️ not measured |

The harness also measures **time spent waiting on transfers**, using CUDA events (timestamps recorded by the GPU itself). This is what makes the bottleneck analysis below possible.

### Deliverable 3: Characterizing the bottleneck

#### Where does the time go?

Decode time split into **waiting on expert transfers** and **everything else** (expert math, attention, router, Python overhead):

| Configuration | GPU memory | Decode tok/s | Waiting on transfers | Everything else | Share waiting |
|---|---|---|---|---|---|
| 0 slots/layer | 0.97 GiB | 7.9 | 67.5 s | 60.6 s | **53%** |
| 8 slots/layer | 2.44 GiB | 9.5 | 47.2 s | 59.3 s | **44%** |
| 16 slots/layer | 3.95 GiB | 11.3 | 31.6 s | 58.3 s | **35%** |
| 32 slots/layer | 6.94 GiB | 14.6 | 12.4 s | 57.3 s | **18%** |
| 64 slots/layer | 12.94 GiB | 17.7 | 0.8 s | 56.5 s | **1%** |
| ours, preloaded | 12.94 GiB | 17.8 | 0 s | 57.0 s | 0% |
| GPU-resident (original model) | 12.95 GiB | 30.2 | 0 s | 33.6 s | 0% |

**Prefill** is different: about 4.8 s in every offloaded configuration, with **61%** spent waiting on transfers, compared with 0.39 s for the original model. The cache size made no difference (0 prefill hits in every configuration).

#### How does hit rate degrade as the cache shrinks?

| Slots per layer | Decode hit rate | Drop from the next size up |
|---|---|---|
| 64 | 98.8% | – |
| 32 | 81.3% | −17.5 points |
| 16 | 52.4% | −28.9 points |
| 8 | 28.9% | −23.5 points |
| 0 | 0% | −28.9 points |

The hit rate falls slowly at first, then quickly once the cache drops below 32 slots. The reason is the **working set**: over one prompt, a layer ends up using about 57 of its 64 experts. Once the cache is much smaller than that, useful experts keep getting evicted before they're needed again.

#### Conclusions

**1. The bottleneck depends on cache size.** With a small cache (16 slots or fewer), transfers take 35–53% of decode time. With a large cache (32 slots or more), transfers drop to 18% or less, and the main cost becomes the expert computation itself.

**2. Each miss costs a fixed amount, so hit rate translates directly into speed.** Transfers ran at a steady ~24.5 GB/s in every configuration, so each miss costs about 0.5 ms. The "everything else" time stayed nearly constant (56–61 s). Speed is set almost entirely by the number of misses.

**3. This sets a ceiling on what a better cache policy can achieve.** Even a perfect policy, with zero misses, can only remove the transfer time. At 16 slots, that would take decode from 11.3 tok/s to about 17.4 tok/s, a **54% maximum improvement**. At 8 slots, 9.5 → about 17.1 tok/s, an **80% maximum improvement**. These are the targets future policies are measured against.

**4. Policy work matters most in the 8–32 slot range.** That's where hit rate is most sensitive to cache size, and where smarter eviction has the most room to help. At 64 slots, there's nothing left to gain.

**5. Prefill needs a different fix.** Each prompt starts with an empty cache, so prefill loads every expert for the first time, and no eviction policy can help. Improving prefill requires **prefetching** (loading experts before they're needed) or keeping the cache warm between prompts.

**6. Removing all transfers still leaves a gap to the original model.** Our code tops out at 17.8 tok/s versus 30.2, because it computes experts one at a time in a loop, while the original uses one optimized operation (`grouped_mm`). This matters more for OLMoE than for larger models like Mixtral, because OLMoE's experts are small (12.6 MB vs. about 350 MB), so per-expert overhead is a bigger share of the cost. It also explains why transfers are at most 53% of time here, compared with the 85–95% that HOBBIT reports for Mixtral.

### Grading criteria

**Reproducibility**

- ✅ Fixed workload: `prompts.json`, plus saved tokens in `reference_tokens.json`, so every run processes identical text.
- ✅ Each prompt starts with an empty cache, so prompt order doesn't affect results.
- ✅ Raw results are saved to CSV.
- ✅ Library versions are pinned in `requirements.txt`.
- ✅ The counts (hits, misses, bytes, agreement) were identical across two separate runs, and timings varied by only about 1–2%.
- ⚠️ No setup script yet. A fresh Ubuntu machine also needs `pip install --upgrade pip` before installing requirements.
- ⚠️ Each configuration was run only once. Several runs per configuration would give error bars on the timings.

**Correctness of the methodology**

The measurements were checked against each other and against expectations:

- Every offloaded run agrees with the reference identically (99.02%), so the cache never changes results.
- GPU memory matches the formula exactly: 1.5 GiB more for every 8 slots per layer.
- Transfer speed is the same (~24.5 GB/s) in every configuration, close to what PCIe Gen4 delivers in practice.
- Non-transfer time is nearly constant across configurations and matches the preloaded run, which has no transfers at all. This confirms that the transfer measurement captures the full cost of offloading.
- The comparison uses a fair "no transfers" reference (our code, preloaded), not just the original model. The original uses a faster kernel, and comparing against it alone would overstate how much time goes to transfers.

**Sound empirical conclusion**

The baseline is **transfer-bound with small caches and compute-bound with large ones**, with the crossover around 32 slots per layer. Prefill is transfer-bound at every cache size. This tells the rest of the project where to focus: smarter eviction for decode in the 8–32 slot range, prefetching for prefill, and a faster expert kernel to raise the ceiling.

### Remaining gaps

- Measure CPU memory directly instead of calculating it.
- Add a setup script that installs everything from scratch.
- Run each configuration 3–5 times and report averages with error bars.
