"""RAG retrieval pipeline with relevance scoring and source attribution."""
import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class FinanceRAGRetriever:
    """Retrieves relevant financial knowledge with source attribution."""

    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        logger.info("FinanceRAGRetriever initialized")

    def retrieve(self, query: str, k: int = 5, category: Optional[str] = None) -> List[Document]:
        """Retrieve relevant documents for a query."""
        try:
            if category:
                # Filter by category using metadata filter
                docs = self.vectorstore.similarity_search(
                    query, k=k * 2,  # get more then filter
                )
                docs = [d for d in docs if d.metadata.get("category") == category][:k]
            else:
                docs = self.vectorstore.similarity_search(query, k=k)

            logger.debug(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Retrieve documents with similarity scores."""
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"RAG retrieval with scores failed: {e}")
            return []

    def format_context(self, documents: List[Document], include_sources: bool = True) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not documents:
            return "No relevant background information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Financial Knowledge")
            source = doc.metadata.get("source", "Financial Education Series")
            content = doc.page_content

            if include_sources:
                context_parts.append(f"[{i}] {title}\n{content}\nSource: {source}")
            else:
                context_parts.append(content)

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """Extract source attribution from documents."""
        sources = []
        seen = set()
        for doc in documents:
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")
            key = f"{title}_{source}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "title": title,
                    "source": source,
                    "category": doc.metadata.get("category", "general")
                })
        return sources
