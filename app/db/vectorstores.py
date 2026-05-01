import os
import shutil
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import logger
from app.core.settings import (
    CORE_KNOWLEDGE_COLLECTION,
    CORE_KNOWLEDGE_DIR,
    USER_MEMORY_COLLECTION,
    USER_MEMORY_DIR,
)
from app.llm.client import embeddings


def ingest_knowledge(
    persist_dir: str,
    collection_name: str,
    doc_folder: str | None = None,
    force_rebuild: bool | None = False,
):
    logger.info(f"[System]: Start {collection_name} Knowledge base creation")
    try:
        if force_rebuild and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

        if doc_folder and os.path.exists(doc_folder) and collection_name == CORE_KNOWLEDGE_COLLECTION:
            loader = DirectoryLoader(doc_folder, glob="**/*.txt")
            documents = loader.load()

            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                )
                chunks = text_splitter.split_documents(documents)

                for chunk in chunks:
                    source = chunk.metadata.get("source", "").lower()
                    if "recipes" in source:
                        category = "recipe"
                    elif "nutrition" in source:
                        category = "nutrition"
                    elif "safety" in source:
                        category = "safety"
                    else:
                        category = "general"
                    chunk.metadata = {
                        "category": category,
                        "tags": [],
                        "timestamp": datetime.now().isoformat(),
                        "source": "core",
                    }

                vectorstore.add_documents(chunks)

        logger.info(f"[System]:✅ {collection_name} Knowledge base created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"[System]: ❌ Error while creating {collection_name} Knowledge base:{str(e)}")


knowledge_vectorstore = ingest_knowledge(
    persist_dir=CORE_KNOWLEDGE_DIR,
    collection_name=CORE_KNOWLEDGE_COLLECTION,
)
user_vectorstore = ingest_knowledge(
    persist_dir=USER_MEMORY_DIR,
    collection_name=USER_MEMORY_COLLECTION,
)


def runtime_context():
    return {
        "knowledge_vectorstore": knowledge_vectorstore,
        "user_vectorstore": user_vectorstore,
    }
