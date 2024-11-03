# tup-rag

An implementation and comparison of different vector databases for retrieval-augmented generation

## Chroma

### Pull the Docker container

```sh
docker pull chromadb/chroma
```

### Start/stop the Docker container

```sh
docker run -p 8000:8000 chromadb/chroma
```

---

## Milvus

### Start/stop the Docker container

```sh
bash milvus_docker.sh start

bash milvus_docker.sh stop

bash milvus_docker.sh delete

bash milvus_docker.sh upgrade
```
