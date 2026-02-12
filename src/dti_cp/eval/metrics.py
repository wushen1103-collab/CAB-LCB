from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class _Fenwick:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.bit = np.zeros(self.n + 1, dtype=np.int64)

    def add(self, idx: int, delta: int) -> None:
        i = int(idx) + 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum_prefix(self, idx: int) -> int:
        if idx < 0:
            return 0
        i = int(idx) + 1
        s = 0
        while i > 0:
            s += int(self.bit[i])
            i -= i & -i
        return s


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Efficient concordance index in O(n log n).
    Counts concordant/discordant/tied pairs among pairs with different y_true.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = y_true.shape[0]
    if n <= 1:
        return float("nan")

    order = np.argsort(y_true, kind="mergesort")
    yt = y_true[order]
    yp = y_pred[order]

    uniq_pred = np.unique(yp)
    ranks = np.searchsorted(uniq_pred, yp).astype(np.int64)
    m = int(len(uniq_pred))
    bit = _Fenwick(m)

    concordant = 0.0
    discordant = 0.0
    ties = 0.0
    processed = 0

    boundaries = np.flatnonzero(np.diff(yt)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))

    for s, e in zip(starts, ends):
        grp_ranks = ranks[s:e]
        for r in grp_ranks:
            less = bit.sum_prefix(int(r) - 1)
            leq = bit.sum_prefix(int(r))
            eq = leq - less
            greater = processed - leq

            concordant += float(less)
            ties += float(eq)
            discordant += float(greater)

        if grp_ranks.size > 0:
            ur, cnt = np.unique(grp_ranks, return_counts=True)
            for rr, cc in zip(ur, cnt):
                bit.add(int(rr), int(cc))

        processed += int(e - s)

    permissible = concordant + discordant + ties
    if permissible == 0.0:
        return float("nan")
    return float((concordant + 0.5 * ties) / permissible)
