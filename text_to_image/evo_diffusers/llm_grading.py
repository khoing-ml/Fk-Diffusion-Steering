"""Evo-side shim for the Gemini LLM grader."""

try:
    from fkd_diffusers.llm_grading import LLMGrader
except ModuleNotFoundError:
    from text_to_image.fkd_diffusers.llm_grading import LLMGrader

__all__ = ["LLMGrader"]
