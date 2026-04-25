"""
Vertex AI deployment step — run after setup_vertex.py training completes.

  python deploy_vertex.py

Deploys 4 trained models to endpoints.
Writes VERTEX_AI_ENDPOINT_* to .env automatically.
Takes ~15-20 min.
"""
import os
import sys

PROJECT_ID = "project-6bf0badc-9510-4a48-9e6"
REGION     = "us-central1"
MACHINE    = "n1-standard-4"
ENV_FILE   = os.path.join(os.path.dirname(__file__), ".env")


def deploy_model(display_name: str) -> str:
    from google.cloud import aiplatform

    models = aiplatform.Model.list(
        filter=f'display_name="{display_name}"',
        project=PROJECT_ID,
        location=REGION,
        order_by="create_time desc",
    )

    if not models:
        raise RuntimeError(
            f"No trained model: '{display_name}'. "
            f"Check https://console.cloud.google.com/vertex-ai/training?project={PROJECT_ID}"
        )

    model = models[0]
    print(f"[deploy] Found: {display_name} ({model.resource_name})")

    # Reuse existing endpoint if already deployed
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{display_name}-endpoint"',
        project=PROJECT_ID,
        location=REGION,
    )
    if endpoints:
        eid = endpoints[0].resource_name.split("/")[-1]
        print(f"[deploy] Already exists: endpoint {eid}")
        return eid

    endpoint = model.deploy(
        deployed_model_display_name=display_name,
        machine_type=MACHINE,
        min_replica_count=1,
        max_replica_count=1,
        traffic_split={"0": 100},
    )

    eid = endpoint.resource_name.split("/")[-1]
    print(f"[deploy] Done: {display_name} → endpoint {eid}")
    return eid


def write_env(ids: dict):
    lines = []
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            lines = f.readlines()

    updated = set()
    new_lines = []
    for line in lines:
        matched = False
        for key, val in ids.items():
            if line.startswith(f"{key}="):
                new_lines.append(f"{key}={val}\n")
                updated.add(key)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for key, val in ids.items():
        if key not in updated:
            new_lines.append(f"{key}={val}\n")

    with open(ENV_FILE, "w") as f:
        f.writelines(new_lines)

    print(f"\n[env] Written to {ENV_FILE}:")
    for k, v in ids.items():
        print(f"  {k}={v}")


def main():
    print("=" * 60)
    print("Auditra — Vertex AI Deploy (4 models)")
    print(f"Project : {PROJECT_ID}")
    print(f"Region  : {REGION}")
    print("=" * 60)

    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION)

    models_to_deploy = {
        "VERTEX_AI_ENDPOINT_COMPAS":      "auditra-chain-scorer-compas",
        "VERTEX_AI_ENDPOINT_ADULT_TRAIN": "auditra-chain-scorer-adult-train",
        "VERTEX_AI_ENDPOINT_ADULT_TEST":  "auditra-chain-scorer-adult-test",
        "VERTEX_AI_ENDPOINT_GERMAN":      "auditra-chain-scorer-german",
    }

    endpoint_ids = {}
    for env_key, display_name in models_to_deploy.items():
        try:
            endpoint_ids[env_key] = deploy_model(display_name)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            endpoint_ids[env_key] = ""

    write_env(endpoint_ids)

    print("\n" + "=" * 60)
    print("Deployment complete. Restart server:")
    print("  uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
