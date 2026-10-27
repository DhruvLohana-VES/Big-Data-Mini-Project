"""
LLM Annotation and Prompt Engineering Module for Code-Mixed Indian Social Text.
"""
from .prompts import (
    get_zero_shot_prompt,
    get_few_shot_prompt,
    get_reasoning_aware_prompt,
    SYSTEM_INSTRUCTION
)
from .annotator import LLMAnnotator, annotate_dataset
from .evaluator import LLMEvaluator, evaluate_llm_vs_gold

__all__ = [
    "get_zero_shot_prompt",
    "get_few_shot_prompt",
    "get_reasoning_aware_prompt",
    "SYSTEM_INSTRUCTION",
    "LLMAnnotator",
    "annotate_dataset",
    "LLMEvaluator",
    "evaluate_llm_vs_gold"
]
