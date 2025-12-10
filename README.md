# ASK_MARDI: Embedding Updater

This project creates / updates the embeddings for the ASK_MARDI service.

## Installation

- Clone the Git repository
- Create a virtual environment
- Install the dependencies: `pip install -r requirements.txt`

#### Dependencies
- Qdrant server
- lakeFS server

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

# How to run

## Locally - Standalone

- Adjust config.yaml (copy from config_example.yaml)
- Run `python workflow_main.py`


## Locally - Docker

#### Build the image
```
docker build -f docker/Dockerfile -t ghcr.io/mardi4nfdi/askmardi_embedding_updater:dev .
```


#### Run the image
```
docker run --rm \
  -e LAKEFS_USER=your-user \
  -e LAKEFS_PASSWORD=your-pass \
  -e QDRANT_URL=https://your-qdrant.example.com:6333 \
  ghcr.io/mardi4nfdi/askmardi_embedding_updater:dev
```



## Running on a self-hosted Prefect Server

#### Prepare Your Prefect Server Environment (ONLY ONCE)
- Create work-pool 
   - `prefect work-pool create K8WorkerPool --type kubernetes`
   - Hint: The Prefect Work Pool is not a resource running in Kubernetes; it is a metadata object on the Prefect Server.
   - Assumption: there is a Kubernetes pod running `prefect worker start --pool "K8WorkerPool"`

#### Prepare Your Local Environment (ONLY ONCE)
- Set environment variables
   - `prefect config set PREFECT_API_URL="http://your-server/api"`
   - `$env:PREFECT_API_AUTH_STRING="admin:supersecret"`
- Check it is working
   - `prefect deployment ls`

#### Deploy and Run 
- Deploy: `python .\workflow_deploy_kubernetes.py`
- Run: Go to the web ui -> _Deployments_ -> Run the workflow
