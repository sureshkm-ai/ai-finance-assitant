"""Tests for RAG system components."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import json

class TestKnowledgeBaseData:
    def test_articles_file_exists(self):
        articles_path = Path(__file__).parent.parent / "src/data/articles/basics.json"
        assert articles_path.exists(), "basics.json articles file should exist"

    def test_articles_are_valid_json(self):
        articles_path = Path(__file__).parent.parent / "src/data/articles/basics.json"
        with open(articles_path) as f:
            data = json.load(f)
        assert "articles" in data
        assert len(data["articles"]) >= 10

    def test_article_structure(self):
        articles_path = Path(__file__).parent.parent / "src/data/articles/basics.json"
        with open(articles_path) as f:
            data = json.load(f)
        for article in data["articles"]:
            assert "id" in article
            assert "title" in article
            assert "content" in article
            assert "category" in article
            assert len(article["content"]) > 100, f"Article {article['id']} content too short"

    def test_glossary_exists(self):
        glossary_path = Path(__file__).parent.parent / "src/data/glossary.json"
        assert glossary_path.exists()

    def test_glossary_has_key_terms(self):
        glossary_path = Path(__file__).parent.parent / "src/data/glossary.json"
        with open(glossary_path) as f:
            data = json.load(f)
        terms = [t["term"].lower() for t in data["terms"]]
        important_terms = ["stock", "bond", "etf", "diversification", "compound interest"]
        for term in important_terms:
            assert any(t in term for t in terms) or any(term in t for t in terms), \
                f"Important term '{term}' not found in glossary"

class TestRetrieverInterface:
    def test_retriever_has_required_methods(self):
        from src.rag.retriever import FinanceRAGRetriever
        from unittest.mock import MagicMock
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = []
        retriever = FinanceRAGRetriever(mock_vectorstore)
        assert hasattr(retriever, "retrieve")
        assert hasattr(retriever, "format_context")
        assert hasattr(retriever, "get_sources")

    def test_format_context_empty(self):
        from src.rag.retriever import FinanceRAGRetriever
        from unittest.mock import MagicMock
        mock_vectorstore = MagicMock()
        retriever = FinanceRAGRetriever(mock_vectorstore)
        result = retriever.format_context([])
        assert "No relevant" in result
