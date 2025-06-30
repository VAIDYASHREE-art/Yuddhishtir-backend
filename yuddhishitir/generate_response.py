# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 17:18:04 2025

@author: DELL
"""

# yuddhishtir/generate_response.py

from .rag_system import YuddhishtirRAGSystem
from .prompt_utils import build_rag_prompt
from .config import PROCESSED_DATA_PATH

rag_system = YuddhishtirRAGSystem(PROCESSED_DATA_PATH)

def generate_response(user_query: str) -> str:
    results = rag_system.semantic_search(user_query, top_k=5)

    if results:
        context_snippets = [r['document']['text'] for r in results]
        return build_rag_prompt(user_query, context_snippets)
    else:
        return "Sorry, I couldn’t find any info in the current knowledge base. Try rephrasing your question or ask about a specific doctor or procedure."