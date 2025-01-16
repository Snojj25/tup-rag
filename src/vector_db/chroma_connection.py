# import chromadb
from chromadb import HttpClient, Documents, EmbeddingFunction, Embeddings
import uuid

from utils.embedder import embedd_sequences

chroma_client = HttpClient(host="localhost", port=8000)
chroma_client.heartbeat()


class MyEmbeddingFunction(EmbeddingFunction):
    emb_func = None

    def __init__(self, func):
        self.emb_func = func

    def __call__(self, input: Documents) -> Embeddings:
        return self.emb_func(input)


def get_collection(collection_name, embedding_func=embedd_sequences):

    if embedding_func:
        my_embedding_function = MyEmbeddingFunction(embedding_func)

        collection = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=my_embedding_function
        )
    else:
        collection = chroma_client.get_or_create_collection(name=collection_name)

    return collection


def insert_documents_chroma(collection, documents=None, embeddings=None):

    if (embeddings == None) == (documents == None):
        raise "Either documents or embeddings must be provided, but not both."

    ids = _generate_uuid_ids(len(documents) if documents else len(embeddings))

    collection.upsert(documents=documents, embeddings=embeddings, ids=ids)


def get_documents_chroma(collection, query_texts=None, query_embeddings=None, top_k=1):

    if (query_texts == None) == (query_embeddings == None):
        raise "Either query_texts or query_embeddings must be provided, but not both."

    results = collection.query(
        query_texts=query_texts,
        query_embeddings=query_embeddings,
        n_results=top_k,
    )

    return results


def delete_collection_chroma(collection_name):
    chroma_client.delete_collection(name=collection_name)


def _generate_uuid_ids(n):
    """Generate array of n UUID-based IDs"""
    return [str(uuid.uuid4()) for _ in range(n)]


# Example with timing
import time


def benchmark_search(collection, query, k=3, num_searches=100):
    start_time = time.time()

    for _ in range(num_searches):
        results = collection.query(query_texts=[query], n_results=k)

    end_time = time.time()
    avg_time = (end_time - start_time) / num_searches

    print(f"Average search time: {avg_time:.4f} seconds")
    return results


import numpy as np

# if __name__ == "__main__":

#     # Create some random vectors and metadata
#     num_vectors = 10000
#     vectors = (
#         np.random.random((num_vectors, 128)).astype(np.float32).tolist()
#     )  # Ensure float32 type

#     collection = get_collection("random_vectors")

#     # Insert vectors
#     t1 = time.time()
#     insert_documents_chroma(collection, embeddings=vectors)
#     print(f"Inserted {num_vectors} vectors in {time.time() - t1:.2f}s")

#     # Perform a search
#     t1 = time.time()
#     query = np.random.random(128).astype(np.float32).tolist()  # Ensure float32 type
#     for _ in range(10):
#         results = get_documents_chroma(collection, query_embeddings=query, top_k=5)
#     print(f"Search time: {time.time() - t1:.2f}s")

#     print("\nSearch results:")
#     for i in range(len(results["ids"])):
#         print(f"ID: {results["ids"][i]}, Distance: {results["distances"][i]}")

#     chroma_client.delete_collection(name="random_vectors")

#     #

#     #

#     #

#     #

#     # documents = [
#     #     "Polar bears are very dangerous",
#     #     "Polar bears kill and eat seals and penguins",
#     #     "You should not approach a dinasour in the wild",
#     # ]

#     # insert_documents_chroma("polar_bears", documents)

#     # results = get_documents_chroma("polar_bears", "polar bears", top_k=3)

#     # print(results)

#     # # collection = _get_collection("polar_bears")
#     # # print(collection.peek(15)["documents"])

#     # chroma_client.delete_collection(name="polar_bears")


delete_collection_chroma("test_rag")
