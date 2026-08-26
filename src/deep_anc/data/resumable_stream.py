"""worker 수와 prefetch에 무관한 global-index 학습 스트림."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def indexed_rng(
    seed: int,
    stream_id: int,
    global_item_index: int,
    attempt: int = 0,
) -> np.random.Generator:
    """순차 RNG 상태 대신 절대 item index로 난수를 유도한다."""

    if int(global_item_index) < 0 or int(attempt) < 0:
        raise ValueError("global_item_index/attempt는 0 이상이어야 합니다")
    sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            int(stream_id) & 0xFFFFFFFF,
            int(global_item_index) & 0xFFFFFFFF,
            int(global_item_index) >> 32,
            int(attempt) & 0xFFFFFFFF,
        ]
    )
    return np.random.default_rng(sequence)


def worker_global_item_indices(
    *,
    start_batch_index: int,
    batch_size: int,
    worker_id: int,
    num_workers: int,
) -> Iterator[int]:
    """DataLoader worker가 담당할 절대 item index를 batch 순서로 반환."""

    start = int(start_batch_index)
    size = int(batch_size)
    worker = int(worker_id)
    workers = int(num_workers)
    if start < 0 or size < 1 or workers < 1 or not 0 <= worker < workers:
        raise ValueError("global stream batch/worker 계약 위반")
    batch = start + worker
    while True:
        base = batch * size
        for offset in range(size):
            yield base + offset
        batch += workers


__all__ = ["indexed_rng", "worker_global_item_indices"]
