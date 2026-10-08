"""Stage 4b: timing helpers.

Two kinds of time:
  - Wall-clock time of a phase (prefill or decode), via now().
  - GPU time spent copying experts, via TransferTimer (CUDA events).
"""
import time
from contextlib import contextmanager

import torch


def now():
    """Wall-clock time after all queued GPU work has finished.

    GPU work runs asynchronously: Python can be far ahead of the GPU. Without
    the synchronize, a timer would stop before the GPU had done the work.
    """
    torch.cuda.synchronize()
    return time.perf_counter()


class TransferTimer:
    """Measures GPU time spent on expert copies.

    A CUDA event is a timestamp recorded *by the GPU* when it reaches that point
    in its queue. An event before and after each copy gives the copy's real
    duration on the GPU, not when Python happened to issue it.

    Assumption: everything runs on ONE CUDA stream, so while a copy runs the GPU
    does nothing else. Copy time is therefore exactly time spent waiting on
    expert loads. Once prefetching copies on a second stream, copies overlap with
    compute and this accounting has to change.
    """

    def __init__(self):
        self.pairs = []

    @contextmanager
    def transfer(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.pairs.append((start, end))

    def collect_ms(self):
        """Total copy time (ms) since the last call. Synchronizes the GPU."""
        torch.cuda.synchronize()
        total = sum(s.elapsed_time(e) for s, e in self.pairs)
        self.pairs.clear()
        return total