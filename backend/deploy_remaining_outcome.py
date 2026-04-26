"""
Deploy remaining 2 outcome-scorer endpoints (adult-test + german).
COMPAS and adult-train already deployed by deploy_outcome_models.py.

Uses n1-standard-2 (2 vCPU) instead of n1-standard-4 to stay within
CustomModelServingCPUsPerProjectPerRegion quota.

Run:
  python deploy_remaining_outcome.py
"""
import os

PROJECT_ID = "project-6bf0badc-9510-4a48-9e6"
REGION     = "us-central1"
MACHINE    = "n1-standard-2"   # 2 vCPU — half quota usage vs n1-standard-4
ENV_FILE   = os.path.join(os.path.dirname(__file__), ".env")

# IDs already deployed successfully by deploy_outcome_models.py
ALREADY_DEPLOYED = {
    "VERTEX_AI_OUTCOME_COMPAS":      "2153407266894643200",
    "VERTEX_AI_OUTCOME_ADULT_TRAIN": "4477827624571240448",
}

# Remaining models to deploy
REMAINING = {
    "VERTEX_AI_OUTCOME_ADULT_TEST": "auditra-outcome-scorer-adult-test",
    "VERTEX_AI_OUTCOME_GERMAN":     "auditra-outcome-scorer-german",
}


def deploy_model(display_name: str) -> str:
    from google.cloud import aiplatform

    models = aiplatform.Model.list(
        filter=f'display_name="{display_name}"',
        project=PROJECT_ID,
        location=REGION,
        order_by="create_time desc",
    )
    if not models:
        raise RuntimeError(f"No trained model: '{display_name}'")

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
    print(f"[deploy] Done: {display_name} → endpoint {eid}  (machine={MACHINE})")
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


def main():
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print("=" * 60)
    print("Auditra — Deploy Remaining 2 Outcome Endpoints")
    print(f"Machine : {MACHINE} (2 vCPU — quota-safe)")
    print("=" * 60)

    # Write already-deployed IDs first
    print("\n[1/3] Writing already-deployed endpoint IDs to .env...")
    write_env(ALREADY_DEPLOYED)
    for k, v in ALREADY_DEPLOYED.items():
        print(f"  {k}={v}")

    # Deploy remaining 2
    print(f"\n[2/3] Deploying {len(REMAINING)} remaining models...")
    deployed = {}
    failed = {}
    for env_key, display_name in REMAINING.items():
        try:
            eid = deploy_model(display_name)
            deployed[env_key] = eid
        except Exception as e:
            print(f"  ERROR {display_name}: {e}")
            failed[env_key] = str(e)
            deployed[env_key] = ""

    # Write all to .env
    print("\n[3/3] Writing to .env...")
    write_env(deployed)

    print("\n" + "=" * 60)
    print("Results:")
    for k, v in {**ALREADY_DEPLOYED, **deployed}.items():
        status = "OK" if v else "FAILED"
        print(f"  [{status}] {k}={v}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for k, e in failed.items():
            print(f"  {k}: {e}")
        print("\nIf quota still exceeded, undeploy chain-scorer endpoints first:")
        print("  → GCP Console > Vertex AI > Endpoints > delete auditra-chain-scorer-* deployments")
        print("  → Then re-run: python deploy_remaining_outcome.py")
    else:
        print("\nAll deployed. Restart server:")
        print("  pkill -f uvicorn")
        print("  nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &")
    print("=" * 60)


if __name__ == "__main__":
    main()
