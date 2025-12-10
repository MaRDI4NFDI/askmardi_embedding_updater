# For execution on a Prefect server the secrets have to be set.
# See README.md for details.

# Run this for CLOUD execution:
#   prefect cloud login

# To add a schedule:
#   * Go to "Deployments"
#   * Click on the workflow name
#   * Click on "+ Schedule" (top right corner)

# Run this for LOCAL execution:
#   prefect config unset PREFECT_API_URL
#   prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
#   prefect server start

from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/MaRDI4NFDI/askmardi_embedding_updater.git",
        entrypoint="workflow_main.py:start_update_embedding_workflow",
    ).deploy(
        name="askmardi_embedding_updater",
        work_pool_name="K8WorkerPool",
        job_variables={
            "image": "ghcr.io/mardi4nfdi/askmardi_embedding_updater:latest",
        },
    )
