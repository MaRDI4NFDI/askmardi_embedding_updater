# ASK_MARDI: Embedding Updater

This project creates / updates the embeddings for the ASK_MARDI service.

## Local Development

#### Qdrant
Make sure you have a qdrant instance running.

```
cd ~
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

#### LakeFS
Currently, lakeFS is used as the storage backend for the documents (PDFs). 


#### Configuration
Create a config.yaml file in the root directory of the project. 
Copy standard values from the provided config_example.yaml and adjust the values.

## Docker

#### Build the image
```
docker build -f docker/Dockerfile -t ghcr.io/mardi4nfdi/askmardi_embedding_updater:dev .
```


#### Run the image
```
docker run --rm \
  -e LAKEFS_USER=your-user \
  -e LAKEFS_PASSWORD=your-pass \
  ghcr.io/mardi4nfdi/askmardi_embedding_updater:latest
```

