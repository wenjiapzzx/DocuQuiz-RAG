"""
DocuQuiz-RAG 文档自动出题系统
基于RAG技术的智能文档题目生成与评估系统
"""

__version__ = "1.0.0"
__author__ = "DocuQuiz Team"
__description__ = "文档自动出题系统 - 基于RAG技术的智能题目生成与质量评估"

from .main import DocuQuizSystem
from .config import initialize_environment, load_embedding_model
from .llm import setup_deepseek_llm
from .document_processor import DocumentProcessor, HybridRetriever
from .quiz_generator import QuizGenerator
from .evaluator import SurveyEvaluator
from .difficulty_analyzer import QuestionDifficultyAnalyzer

__all__ = [
    "DocuQuizSystem",
    "initialize_environment", 
    "load_embedding_model",
    "setup_deepseek_llm",
    "DocumentProcessor",
    "HybridRetriever", 
    "QuizGenerator",
    "SurveyEvaluator",
    "QuestionDifficultyAnalyzer"
]