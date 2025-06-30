# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 17:17:17 2025

@author: DELL
"""

# yuddhishtir/rag_system.py

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class YuddhishtirRAGSystem:
    def __init__(self, processed_data_path: str):
        self.rag_documents = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.load_processed_data(processed_data_path)

    def load_processed_data(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.rag_documents = data['documents']
        self._build_search_index()

    def _build_search_index(self):
        texts = [doc['text'] for doc in self.rag_documents]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def semantic_search(self, query: str, top_k: int = 5):
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'document': self.rag_documents[idx],
                    'similarity_score': float(similarities[idx]),
                    'rank': len(results) + 1
                })
        return results