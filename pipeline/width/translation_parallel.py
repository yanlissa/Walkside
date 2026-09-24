from __future__ import annotations

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def _evaluate_candidate_batch(
    edge_line, axis_segments, axis_data, indexed_candidates,
    edge_step, max_match_distance, min_parallel_cos, shift_length_weight,
):
    from pipeline.width.skeleton import _evaluate_scored_translation_candidate

    best = None
    best_score = -1e+18
    for candidate_index, candidate in indexed_candidates:
        item = _evaluate_scored_translation_candidate(
            edge_line, axis_segments, axis_data, candidate,
            edge_step, max_match_distance, min_parallel_cos, shift_length_weight,
        )
        if item is not None and item['global_score'] > best_score:
            best_score = item['global_score']
            best = (candidate_index, item)
    return best


def _prefer_result(best, result):
    if result is None:
        return best
    if best is None:
        return result
    index, item = result
    best_index, best_item = best
    score, best_score = item['global_score'], best_item['global_score']
    if score > best_score or (score == best_score and index < best_index):
        return result
    return best


class TranslationCandidatePool:
    def __init__(self, workers: int = 2):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer (1 = sequential).')
        if sys.platform == 'win32' and workers > 61:
            raise ValueError('ProcessPoolExecutor supports at most 61 workers on Windows.')
        self.workers = workers
        self._executor: ProcessPoolExecutor | None = None
        self._active = False
        self._closed = False
        self.searches_submitted = 0
        self.candidates_submitted = 0
        self.batches_submitted = 0

    def __enter__(self) -> 'TranslationCandidatePool':
        if self._active or self._closed:
            raise RuntimeError('Create a fresh TranslationCandidatePool for each width pass.')
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=exc_type is not None)
        finally:
            self._active = False
            self._closed = True
            self._executor = None
        return False

    def should_parallelize(self, candidate_count: int) -> bool:
        if not self._active:
            raise RuntimeError('Use TranslationCandidatePool inside a with block.')
        return self.workers > 1 and candidate_count > 1

    def evaluate_best(
        self, *, edge_line, axis_segments, axis_data, candidates,
        edge_step, max_match_distance, min_parallel_cos, shift_length_weight,
    ) -> dict[str, Any] | None:
        if not self.should_parallelize(len(candidates)):
            raise RuntimeError('Small candidate lists must use the sequential path.')
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context('spawn'),
            )

        indexed = list(enumerate(candidates))
        batch_size = max(1, (len(indexed) + 2 * self.workers - 1) // (2 * self.workers))
        futures = []
        best = None
        try:
            for start in range(0, len(indexed), batch_size):
                futures.append(self._executor.submit(
                    _evaluate_candidate_batch,
                    edge_line, axis_segments, axis_data,
                    indexed[start:start + batch_size],
                    edge_step, max_match_distance, min_parallel_cos, shift_length_weight,
                ))
            self.searches_submitted += 1
            self.candidates_submitted += len(indexed)
            self.batches_submitted += len(futures)
            for future in as_completed(futures):
                best = _prefer_result(best, future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return None if best is None else best[1]
