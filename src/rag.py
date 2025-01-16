from typing import Dict, List

import numpy
import torch

from inference import (
    optimize_memory,
    setup_tokenizer_and_model,
    get_model_output,
)
from utils.chunkify_docx import DocxChunker
from utils.db_types import VectorDBType
from utils.embedder import embedd_sequences
from vector_db.chroma_connection import (
    insert_documents_chroma,
    get_documents_chroma,
    delete_collection_chroma,
)

# from utils import DocxChunker, VectorDBType, embedd_sequences ??? Can we do this?
from vector_db.milvus_connection import SimpleVectorDB


def chunkify_document(
    docx_path: str, chunk_size: int = 200, chunk_overlap: int = 40
) -> List[str]:
    chunker = DocxChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.process_docx(docx_path)

    return chunks


def _embed_chunks(chunks: List[str]) -> numpy.ndarray:
    embeddings = embedd_sequences(chunks)

    return embeddings.numpy()


def store_embedding_vectors_in_vector_db(
    chunks: List[str], db_type: VectorDBType
) -> None:

    if db_type == VectorDBType.MILVUS:
        # Store vectors in Milvus

        # Initialize the vector database
        db = SimpleVectorDB()

        embedding_vectors = _embed_chunks(chunks)

        # Insert vectors and metadata
        db.insert(embedding_vectors, chunks)
        print(f"Inserted {len(chunks)} vectors")

        db.close()

    elif db_type == VectorDBType.CHROMA:
        # Store vectors in Chroma

        insert_documents_chroma(collection_name="test_rag", documents=chunks)

    else:
        raise ValueError(f"Unsupported vector DB type: {db_type}")


QueryResult = Dict[str, List[str] | List[float]]


def _handle_chroma_query_results(results) -> QueryResult:
    ids, distances, documents = (
        results["ids"],
        results["distances"],
        results["documents"],
    )

    return {"ids": ids, "distances": distances, "documents": documents}


def query_vector_db(query: str, db_type: VectorDBType, top_k: int = 3) -> QueryResult:

    if db_type == VectorDBType.MILVUS:
        # Store vectors in Milvus

        # Initialize the vector database
        db = SimpleVectorDB()

        query_embedding = _embed_chunks([query])[0]

        # Perform a search
        results = db.search(query_embedding, top_k=top_k)

        # Close connection
        db.close()

        return results

    elif db_type == VectorDBType.CHROMA:
        # Store vectors in Chroma

        results = get_documents_chroma(
            collection_name="test_rag", query=query, top_k=top_k
        )

        return _handle_chroma_query_results(results)
    else:
        raise ValueError(f"Unsupported vector DB type: {db_type}")


def clean_response(response: str, original_prompt: str) -> str:
    # if "[end]" in response:
    #     return response.split("[end]", 1)[1].strip()

    if response.startswith(original_prompt):
        return response[len(original_prompt) :].strip().lstrip(".")

    return response


def run_inference(question: str, context: str = None) -> str:

    prompt: str
    if context:
        prompt = f"""Anwser the following question: "{question}". You can use the following context if it's relevant: "{context}"."""
    else:
        prompt = f"""Anwser the following question: "{question}"."""

    optimize_memory()

    tokenizer, model = setup_tokenizer_and_model()

    try:
        response = get_model_output(prompt=prompt, model=model, tokenizer=tokenizer)

        response = clean_response(response, original_prompt=prompt)

        return response
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            print("GPU out of memory, try clearing the cache...")
            raise e
        else:
            raise e


if __name__ == "__main__":

    user_query = (
        "What are the rules on extradition in Slovenia as stated in the constitution?"
    )
    response = run_inference(user_query)
    print("Response without rag: ", response)

    # Example usage
    docx_path = "media/Constitution.docx"
    chunks = chunkify_document(docx_path)

    store_embedding_vectors_in_vector_db(chunks, db_type=VectorDBType.CHROMA)

    results = query_vector_db(user_query, db_type=VectorDBType.CHROMA, top_k=2)

    response = run_inference(user_query, context=results["documents"][0])
    print("Response with rag: ", response)
