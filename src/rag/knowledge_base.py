"""Knowledge base builder: loads financial articles and creates FAISS vector index."""
import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 22MB local model, no API key needed


def load_articles_from_json(filepath: Path) -> List[Document]:
    """Load financial education articles from JSON file."""
    documents = []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        for article in data.get("articles", []):
            doc = Document(
                page_content=article["content"],
                metadata={
                    "id": article.get("id", ""),
                    "title": article.get("title", ""),
                    "category": article.get("category", "general"),
                    "tags": ", ".join(article.get("tags", [])),
                    "source": article.get("source", "Financial Education Series"),
                }
            )
            documents.append(doc)
    except Exception as e:
        logger.error(f"Failed to load articles from {filepath}: {e}")
    return documents


def load_glossary(filepath: Path) -> List[Document]:
    """Load financial glossary as documents."""
    documents = []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        for item in data.get("terms", []):
            doc = Document(
                page_content=f"{item['term']}: {item['definition']}",
                metadata={
                    "id": f"glossary_{item['term'].lower().replace(' ', '_')}",
                    "title": item["term"],
                    "category": "glossary",
                    "tags": "glossary, definition, financial terms",
                    "source": "Financial Glossary"
                }
            )
            documents.append(doc)
    except Exception as e:
        logger.error(f"Failed to load glossary from {filepath}: {e}")
    return documents


def build_knowledge_base(force_rebuild: bool = False) -> FAISS:
    """Build or load the FAISS knowledge base from financial education content."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load cached index if exists and not forcing rebuild
    if INDEX_PATH.exists() and not force_rebuild:
        logger.info("Loading existing FAISS index...")
        try:
            vectorstore = FAISS.load_local(
                str(INDEX_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded FAISS index from {INDEX_PATH}")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}. Rebuilding...")

    # Load all documents
    all_documents = []
    articles_dir = DATA_DIR / "articles"

    # Load all JSON article files
    for json_file in articles_dir.glob("*.json"):
        docs = load_articles_from_json(json_file)
        all_documents.extend(docs)
        logger.info(f"Loaded {len(docs)} articles from {json_file.name}")

    # Load glossary
    glossary_file = DATA_DIR / "glossary.json"
    if glossary_file.exists():
        glossary_docs = load_glossary(glossary_file)
        all_documents.extend(glossary_docs)
        logger.info(f"Loaded {len(glossary_docs)} glossary terms")

    if not all_documents:
        raise ValueError("No documents loaded for knowledge base")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = text_splitter.split_documents(all_documents)
    logger.info(f"Split {len(all_documents)} documents into {len(split_docs)} chunks")

    # Create FAISS index
    logger.info("Building FAISS vector index (this may take a minute)...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Save index
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    logger.info(f"FAISS index saved to {INDEX_PATH}")

    return vectorstore


def get_vectorstore() -> FAISS:
    """Get or create the vector store singleton."""
    return build_knowledge_base(force_rebuild=False)
