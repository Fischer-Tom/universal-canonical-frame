from __future__ import annotations

import math
import random
from typing import List


def stratified_sample(
    num_frames: int, k: int, rng: random.Random, jitter: bool = True
) -> List[int]:
    """Pick k indices stratified across [0, num_frames) with optional jitter."""
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    k = max(1, min(k, num_frames))
    step = num_frames / k

    idxs: List[int] = []
    for i in range(k):
        c = (i + 0.5) * step
        if jitter:
            lo = max(0, int(math.floor(c - 0.5 * step)))
            hi = min(num_frames - 1, int(math.ceil(c + 0.5 * step)) - 1)
            idxs.append(rng.randint(lo, max(lo, hi)))
        else:
            idxs.append(int(round(c)))

    idxs = sorted({max(0, min(num_frames - 1, i)) for i in idxs})
    while len(idxs) < k:
        for j in range(num_frames):
            if j not in idxs:
                idxs.append(j)
                idxs.sort()
                break
        else:
            break
    return idxs
