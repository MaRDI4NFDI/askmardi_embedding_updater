# For execution on a Prefect server the secrets have to be set.
# See README.md for details.

# To add a schedule:
#   * Go to "Deployments"
#   * Click on the workflow name
#   * Click on "+ Schedule" (top right corner)

# Run this for LOCAL execution:
#   prefect config unset PREFECT_API_URL
#   prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
#   prefect server start

# MAKE SURE that in the web-ui:
#    "Work Pools" -> "K8 Worker Pool" -> "Base Job Configuration" -> "Image Pull Policy" is set to "ALWAYS"


from workflow_main import start_update_embedding_workflow

if __name__ == "__main__":
    start_update_embedding_workflow.deploy(
        name="askmardi_embedding_updater",
        work_pool_name="K8WorkerPool",
        image="ghcr.io/mardi4nfdi/askmardi_embedding_updater:latest",
        push=False,
        build=False,
        job_variables={
            "imagePullPolicy": "Always",
            "env": {
                "QDRANT_URL": "http://qdrant:6333",
            },
        },
        parameters={
            "update_embeddings_loop_iterations": 10,
            "update_embeddings_embeddings_per_loop": 50,
        },
    )