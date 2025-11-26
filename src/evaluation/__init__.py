"""
Evaluation module for parallel reward function evaluation.

This module provides a complete pipeline for evaluating reward functions
in parallel across a distributed machine pool.

Main classes:
- RewardEvaluator: High-level orchestrator for the evaluation pipeline
- WorkspaceManager: Handles workspace preparation and code injection
- ParallelExecutor: Manages distributed parallel execution
- ResultProcessor: Processes training results and TensorBoard logs

Example usage:
    >>> from src.evaluation import RewardEvaluator
    >>> from src.utils.common import load_machine_pool
    >>>
    >>> machine_pool = load_machine_pool()
    >>> evaluator = RewardEvaluator(
    ...     task_config={'workspace': '...', 'env_cfg_path': '...', 'logs_path': '...'},
    ...     settings_config={'workspace': '...'},
    ...     machine_pool=machine_pool
    ... )
    >>> best_result = evaluator.evaluate(reward_functions)
"""

from .evaluator import RewardEvaluator
from .workspace_manager import WorkspaceManager
from .parallel_executor import ParallelExecutor, TrainingTask, ProcessResult, TaskStatus
from .result_processor import ResultProcessor, EvaluationResult
from . import config

__all__ = [
    'RewardEvaluator',
    'WorkspaceManager',
    'ParallelExecutor',
    'ResultProcessor',
    'EvaluationResult',
    'TrainingTask',
    'ProcessResult',
    'TaskStatus',
    'config',
]

__version__ = '0.1.0'
