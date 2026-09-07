"""
Fitness judgement for evaluated reward candidates.

This is the "judge" half that was split out of the evaluator: given records the
evaluator has already trained and captured, it reads the fixed ``fitness_function``
metric from each one's TensorBoard log, writes it onto the record, and selects
the batch winner. The evaluator stays responsible only for dispatch + capture;
all metric reading and ranking lives here so the scoring policy can change
(e.g. mean-over-seeds, multi-objective) without touching dispatch.
"""

import os
import logging
from math import isfinite
from statistics import fmean, stdev
from typing import List, Optional, Tuple

from .result_processor import load_accumulator
from . import config

logger = logging.getLogger(__name__)


class FitnessScorer:
    """Reads the fitness metric from captured runs and ranks candidates."""

    def __init__(self, fitness_tag: str = config.FITNESS_METRIC):
        # Matched by suffix so whatever scope rl_games/IsaacAlgoObserver prefixes
        # it with (e.g. "Episode/fitness_function") still resolves.
        self.fitness_tag = fitness_tag

    # ---------------------------------------------------------------- scoring
    def score(self, record) -> float:
        """Read fitness from ``record.tb_path``, store it on the record, return it."""
        fitness = self.read_fitness(record.tb_path) if record.tb_path else float("-inf")
        record.fitness = fitness
        return fitness

    def score_all(self, records: List) -> List:
        """Score every captured record in ``records``; returns the same list."""
        for record in records:
            self.score(record)
        return records

    def read_fitness(self, tb_file: Optional[str]) -> float:
        """Return the max value of the fitness metric over training (-inf if absent)."""
        if not tb_file or not os.path.exists(tb_file):
            logger.error(f"TensorBoard file not found: {tb_file}")
            return float("-inf")
        try:
            ea = load_accumulator(tb_file)
            tag = self._resolve_tag(ea, self.fitness_tag)
            if tag is None:
                logger.warning(
                    f"Fitness tag {self.fitness_tag!r} not found. "
                    f"Available: {ea.scalars.Keys()}"
                )
                return float("-inf")
            events = ea.Scalars(tag)
            if not events:
                return float("-inf")
            return float(max(e.value for e in events))
        except Exception as e:  # noqa: BLE001 - TB parsing surfaces many error types
            logger.error(f"Error reading fitness from {tb_file}: {e}")
            return float("-inf")

    def _resolve_tag(self, ea, wanted: str) -> Optional[str]:
        """Resolve ``wanted`` to an actual scalar tag, matching by exact or suffix."""
        keys = ea.scalars.Keys()
        if wanted in keys:
            return wanted
        suffix = wanted.split("/")[-1]
        matches = [k for k in keys if k.split("/")[-1] == suffix or k.endswith(suffix)]
        if matches:
            if len(matches) > 1:
                logger.warning(f"Multiple tags match {wanted!r}: {matches}; using {matches[0]}")
            return matches[0]
        return None

    # ---------------------------------------------------------------- ranking
    @staticmethod
    def aggregate_trials(candidates: List, trials: List) -> List:
        """
        Fold each candidate's repeated trainings into the one fitness that ranks it.

        A candidate trained once is ranked on a single noisy number, so the batch
        winner is partly the best reward and partly the luckiest RL seed. Each
        candidate is instead trained ``repeats`` times on different seeds and
        scored by the **trimmed mean** of those runs: the highest and the lowest
        are discarded and the rest averaged. At the usual ``repeats: 3`` that is
        exactly the middle run. Fewer than three scored trials cannot be trimmed
        (nothing would be left), so they are averaged as they stand.

        The candidate also inherits the artifacts of whichever trial landed
        nearest that aggregate, so the training summary fed back to the LLM comes
        from a run that actually scored what the candidate was ranked on, rather
        than from its luckiest attempt.

        Mutates each candidate in place (``fitness``, ``trial_fitnesses``,
        ``status`` and the captured paths) and returns ``candidates``.
        """
        by_candidate = {}
        for trial in trials:
            by_candidate.setdefault(trial.candidate_tag, []).append(trial)

        for candidate in candidates:
            if not candidate.has_method:
                continue          # generation failed; there was nothing to train
            group = by_candidate.get(candidate.tag, [])
            scored = sorted(
                (t for t in group if isfinite(t.fitness)), key=lambda t: t.fitness
            )
            candidate.trial_fitnesses = [t.fitness for t in scored]

            if not scored:
                candidate.fitness = float("-inf")
                if group:
                    # Carry up why it was never measured, so the history
                    # distinguishes a bad reward from a job that never ran.
                    candidate.status = group[0].status
                    candidate.eval_error = (
                        f"no fitness from {len(group)} trial(s); first: "
                        f"{group[0].eval_error or group[0].status}"
                    )
                logger.warning(f"[{candidate.tag}] no trial produced a fitness")
                continue

            kept = scored[1:-1] if len(scored) >= 3 else scored
            candidate.fitness = fmean(t.fitness for t in kept)
            representative = min(
                scored, key=lambda t: abs(t.fitness - candidate.fitness)
            )
            candidate.status = representative.status
            candidate.log_path = representative.log_path
            candidate.tb_path = representative.tb_path
            candidate.summary_path = representative.summary_path
            logger.info(
                f"[{candidate.tag}] fitness {candidate.fitness:.4f} from "
                f"{len(scored)}/{len(group)} trial(s) "
                f"[{', '.join(f'{t.fitness:.4f}' for t in scored)}]"
            )
        return candidates

    @staticmethod
    def select_best(records: List):
        """
        Mark and return the highest-fitness record in ``records``.

        Considers only records with a finite fitness (i.e. successfully trained
        and scored). Returns None if none qualify. Sets ``selected_best`` on the
        winner and clears it on the rest of the batch.
        """
        scored = [r for r in records if isfinite(r.fitness)]
        for r in records:
            r.selected_best = False
        if not scored:
            logger.warning("No scored candidates to select from")
            return None
        best = max(scored, key=lambda r: r.fitness)
        best.selected_best = True
        logger.info(f"Best candidate: {best}")
        return best

    @staticmethod
    def select_top_k(records: List, k: int) -> List:
        """
        Mark and return the ``k`` highest-fitness records in ``records``.

        This is the pool the next iteration breeds from, and it replaces Eureka's
        greedy "keep only the batch winner" step. Pass the *current pool plus the
        new batch* so a parent stays in the running against its own children: the
        pool's fitness can then never regress, which a greedy hand-off allows
        whenever a whole batch scores below its parent.

        Ranks only records with a finite fitness (successfully trained and
        scored), so failed jobs can never occupy a parent slot. ``survived`` is
        sticky — set here and never cleared — so the persisted history records
        which candidates were ever parents, not just who is in the pool right
        now. ``k`` of 1 reproduces the greedy single-parent loop.
        """
        scored = [r for r in records if isfinite(r.fitness)]
        if not scored:
            logger.warning("No scored candidates to form a pool from")
            return []
        top = sorted(scored, key=lambda r: r.fitness, reverse=True)[: max(1, k)]
        for r in top:
            r.survived = True
        logger.info(
            f"Pool of {len(top)}: "
            + ", ".join(f"{r.tag}={r.fitness:.4f}" for r in top)
        )
        return top

    @staticmethod
    def summarise(records: List) -> Tuple[List[float], float, float]:
        """
        Return ``(values, mean, std)`` over ``records`` — one reward's eval seeds.

        The mean is the score to report: a max over seeds would give the best
        moment of the luckiest seed (each seed's fitness is already a max over
        training events) and discard the variance the seeds were trained to
        measure. Unscored records are left out rather than counted as -inf, so
        ``len(values)`` says how many seeds actually ran. ``std`` is the sample
        std (0.0 for a single seed).
        """
        values = [r.fitness for r in records if isfinite(r.fitness)]
        if not values:
            logger.warning("No scored records to summarise")
            return [], float("-inf"), 0.0
        return values, fmean(values), stdev(values) if len(values) > 1 else 0.0
