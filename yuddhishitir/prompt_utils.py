# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 17:16:01 2025

@author: DELL
"""
# yuddhishtir/prompt_utils.py

def build_rag_prompt(user_query: str, context_snippets: list) -> str:
    return f"""
You are Yuddhishtir, an intelligent hospital assistant.

The user asked:
\"{user_query}\"

Relevant Data:
{chr(10).join(context_snippets)}

Using the data above, provide a helpful and polite answer. If the information is insufficient, clearly mention that and suggest next steps.
"""
