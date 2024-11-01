# import chromadb
from chromadb import HttpClient, Documents, EmbeddingFunction, Embeddings
import uuid

chroma_client = HttpClient(host="localhost", port=8000)
chroma_client.heartbeat()


class MyEmbeddingFunction(EmbeddingFunction):
    emb_func = None

    def __init__(self, func):
        self.emb_func = func

    def __call__(self, input: Documents) -> Embeddings:
        return self.emb_func(input)


def get_collection(collection_name, embedding_func=None):

    if embedding_func:
        my_embedding_function = MyEmbeddingFunction(embedding_func)

        collection = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=my_embedding_function
        )
    else:
        collection = chroma_client.get_or_create_collection(name=collection_name)

    return collection


def insert_documents(collection_name, documents, embedding_func=None):

    collection = get_collection(collection_name, embedding_func)

    ids = generate_uuid_ids(len(documents))

    collection.upsert(documents=documents, ids=ids)


def get_documents(collection_name, query, top_k=1, embedding_func=None):

    collection = get_collection(collection_name, embedding_func)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    return results


def generate_uuid_ids(n):
    """Generate array of n UUID-based IDs"""
    return [str(uuid.uuid4()) for _ in range(n)]


if __name__ == "__main__":

    documents = [
        "Polar bears are very dangerous",
        "Polar bears kill and eat seals and penguins",
        "You should not approach a polar bear in the wild",
    ]

    # insert_documents("polar_bears", documents)

    # results = get_documents("polar_bears", "food", top_k=2)

    # print(results["documents"])

    # collection = get_collection("polar_bears")
    # print(collection.peek(15)["documents"])


# # switch `add` to `upsert` to avoid adding the same documents every time
# print(collection.peek())

chroma_client.delete_collection(name="polar_bears")


# # collection.add(
# #     documents=["pineapples are yellow", "oranges are orange", "Apples are red"],
# #     ids=["id4", "id5", "id6"],
# # )

# print(results)
