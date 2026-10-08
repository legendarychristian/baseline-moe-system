"""Stage 3a: expert cache bookkeeping.

No tensors and no GPU yet. This file only decides *which* experts are cached
and *which slot* each one occupies. Stage 3b attaches real GPU buffers.

Two pieces, kept separate on purpose:
  - ExpertCache: fixed number of slots, tracks key -> slot, counts hits/misses.
  - A policy (LRUPolicy here): decides who gets evicted. Future policies
    (LFU, MRS, HOBBIT-style scores) replace this class and nothing else.
"""
from collections import OrderedDict


class LRUPolicy:
    """Evict the key that was used least recently."""

    def __init__(self):
        self.order = OrderedDict()  # oldest first, most recent last

    def on_hit(self, key):
        self.order.move_to_end(key)

    def on_insert(self, key):
        self.order[key] = None

    def on_evict(self, key):
        del self.order[key]

    def choose_victim(self):
        return next(iter(self.order))  # the oldest entry


class ExpertCache:
    """Maps expert keys to a fixed set of slots."""

    def __init__(self, num_slots, policy):
        self.num_slots = num_slots
        self.policy = policy
        self.slot_of = {}                      # key -> slot index
        self.free_slots = list(range(num_slots))
        self.hits = self.misses = self.evictions = 0

    def lookup(self, key):
        """Return the slot holding key, or None on a miss."""
        slot = self.slot_of.get(key)
        if slot is None:
            self.misses += 1
        else:
            self.hits += 1
            self.policy.on_hit(key)
        return slot

    def insert(self, key):
        """Give key a slot, evicting if full. The caller copies weights into it."""
        if not self.free_slots:
            self.evict()
        slot = self.free_slots.pop()
        self.slot_of[key] = slot
        self.policy.on_insert(key)
        return slot

    def evict(self):
        victim = self.policy.choose_victim()
        self.policy.on_evict(victim)
        self.free_slots.append(self.slot_of.pop(victim))
        self.evictions += 1
        return victim

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


if __name__ == "__main__":
    # Tiny worked example: 3 slots, a short sequence of expert IDs.
    # Try predicting each line before you run it.
    cache = ExpertCache(num_slots=3, policy=LRUPolicy())
    for e in [0, 1, 2, 0, 3, 0, 1]:
        if cache.lookup(e) is not None:
            result = "hit "
        else:
            before = set(cache.slot_of)
            cache.insert(e)
            evicted = before - set(cache.slot_of)
            result = "miss" + (f", evicted {evicted.pop()}" if evicted else "")
        recency = list(cache.policy.order)  # oldest -> newest
        print(f"access {e}: {result:18s} cache (oldest->newest): {recency}")
    print(f"\nhits={cache.hits} misses={cache.misses} "
          f"evictions={cache.evictions} hit rate={cache.hit_rate():.0%}")