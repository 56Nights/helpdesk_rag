"""Retrieval-augmented answering over a corpus of resolved helpdesk tickets."""
from .config import Settings
from .pipeline import Answer, Confidence, HelpdeskRAG

__all__ = ["Settings", "HelpdeskRAG", "Answer", "Confidence"]
