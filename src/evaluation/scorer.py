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
import math
import logging
from math import isfinite
from statistics import fmean, stdev
from typing import List, Optional, Tuple

from .result_processor import load_accumulator
from . import config

logger = logging.getLogger(__name__)

# Valid values for FitnessScorer's scoring_mode.
SCORING_MODE_GLOBAL_MAX = "global_max"   # max fitness over the whole run (default, prior behaviour)
SCORING_MODE_LAST_5PCT = "last_5pct"     # max fitness over just the final 5% of logged points
VALID_SCORING_MODES = (SCORING_MODE_GLOBAL_MAX, SCORING_MODE_LAST_5PCT)


class FitnessScorer:
    """Reads the fitness metric from captured runs and ranks candidates."""

    def __init__(
        self,
        fitness_tag: str = config.FITNESS_METRIC,
        scoring_mode: str = SCORING_MODE_GLOBAL_MAX,
    ):
        # Matched on the final path segment so whatever scope rl_games/
        # IsaacAlgoObserver prefixes it with (e.g. "Episode/fitness_function")
        # still resolves. See `_resolve_tag` for why it is not looser than that.
        self.fitness_tag = fitness_tag

        # global_max: a candidate's score is the single highest fitness_function
        # value it ever logged, anywhere in its run. Rewards peak capability, but
        # a candidate that spiked early or mid-run then declined (noise, or a
        # real later instability) still wins over a candidate that never spiked
        # as high but ended up more stable/better by the time training stopped.
        # last_5pct: restricts that same max to only the final 5% of logged
        # points, so a candidate has to still be near its best *late* in
        # training to score well — favours stability/convergence over an
        # early or mid-run fluke that didn't hold up.
        if scoring_mode not in VALID_SCORING_MODES:
            raise ValueError(
                f"scoring_mode={scoring_mode!r} not recognised; expected one of {VALID_SCORING_MODES}"
            )
        self.scoring_mode = scoring_mode

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
        """Return this run's fitness score, per ``self.scoring_mode`` (-inf if absent)."""
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
            events = ea.Scalars(tag)  # chronological (step) order, as logged
            if not events:
                return float("-inf")
            if self.scoring_mode == SCORING_MODE_LAST_5PCT:
                window_size = max(1, math.ceil(len(events) * 0.05))
                events = events[-window_size:]
            return float(max(e.value for e in events))
        except Exception as e:  # noqa: BLE001 - TB parsing surfaces many error types
            logger.error(f"Error reading fitness from {tb_file}: {e}")
            return float("-inf")

    def _resolve_tag(self, ea, wanted: str) -> Optional[str]:
        """Resolve ``wanted`` to an actual scalar tag, by full tag or final segment.

        The match is deliberately anchored to a whole path segment
        ("Episode/fitness_function" resolves, "Episode/components_fitness_function" does
        not). A looser ``endswith`` would let an LLM-named reward component shadow
        the very metric the reward is scored on — the task layer already prefixes
        components with ``components_``, and this is the other half of that guarantee.
        """
        keys = ea.scalars.Keys()
        if wanted in keys:
            return wanted
        suffix = wanted.split("/")[-1]
        matches = [k for k in keys if k.split("/")[-1] == suffix]
        if matches:
            if len(matches) > 1:
                logger.warning(f"Multiple tags match {wanted!r}: {matches}; using {matches[0]}")
            return matches[0]
        return None

    # ---------------------------------------------------------------- ranking
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
