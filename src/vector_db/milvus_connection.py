from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np


class SimpleVectorDB:
    def __init__(
        self, host="localhost", port="19530", collection_name="simple_vectors"
    ):
        """Initialize connection to Milvus server and set up the collection."""
        self.collection_name = collection_name

        # Connect to Milvus server
        connections.connect(host=host, port=port)

        # Define collection schema
        self.dim = 128  # Vector dimension

        # Define the fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=512),
        ]

        schema = CollectionSchema(fields=fields, description="Simple vector database")

        # Create collection if it doesn't exist
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            self.collection = Collection(self.collection_name, schema)

            # Create an IVF_FLAT index for vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            self.collection.create_index(field_name="vector", index_params=index_params)

    def insert(self, vectors, metadata_list):
        """
        Insert vectors and their metadata into the database.

        Args:
            vectors: numpy array of shape (n, dim) containing the vectors
            metadata_list: list of strings containing metadata for each vector
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata entries must match")

        # Ensure vectors are in the correct format (numpy array)
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)

        # Prepare data for insertion
        data = [
            {"vector": vec, "metadata": meta}
            for vec, meta in zip(vectors, metadata_list)
        ]

        # Insert the data
        self.collection.insert(data)
        self.collection.flush()

    def search(self, query_vector, top_k=5):
        """
        Search for the closest vectors to the query vector.

        Args:
            query_vector: numpy array of shape (dim,) containing the query vector
            top_k: number of closest vectors to return

        Returns:
            List of tuples containing (id, distance, metadata) for the closest vectors
        """
        self.collection.load()

        # Ensure query vector is in the correct format
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
        query_vector = query_vector.astype(np.float32)

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["metadata"],
        )

        # Format results
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(
                    {
                        "id": hit.id,
                        "distance": hit.distance,
                        "metadata": hit.entity.get("metadata"),
                    }
                )

        return search_results

    def delete_by_ids(self, ids):
        """Delete vectors by their IDs."""
        expr = f"id in {ids}"
        self.collection.delete(expr)

    def count(self):
        """Return the number of vectors in the database."""
        return self.collection.num_entities

    def close(self):
        """Close the connection to Milvus server."""
        connections.disconnect("default")


# # Example usage
# if __name__ == "__main__":
#     # Initialize the vector database
#     db = SimpleVectorDB()

#     # Create some random vectors and metadata
#     num_vectors = 1000
#     vectors = np.random.random((num_vectors, 128)).astype(
#         np.float32
#     )  # Ensure float32 type
#     metadata = [f"Vector_{i}" for i in range(num_vectors)]

#     # Insert vectors
#     db.insert(vectors, metadata)
#     print(f"Inserted {num_vectors} vectors")

#     # Perform a search
#     query = np.random.random(128).astype(np.float32)  # Ensure float32 type
#     results = db.search(query, top_k=5)

#     print("\nSearch results:")
#     for result in results:
#         print(
#             f"ID: {result['id']}, Distance: {result['distance']}, Metadata: {result['metadata']}"
#         )

#     # Close connection
#     db.close()
