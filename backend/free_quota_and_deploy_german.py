"""
Free CPU quota by undeploying adult-test chain-scorer (code routes adult_test → adult_train),
then deploy german outcome-scorer.

Run:
  python free_quota_and_deploy_german.py
"""
import os

PROJECT_ID = "project-6bf0badc-9510-4a48-9e6"
REGION     = "us-central1"
MACHINE    = "n1-standard-2"
ENV_FILE   = os.path.join(os.path.dirname(__file__), ".env")

from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=REGION)

# ── Step 1: Undeploy model from adult-test chain-scorer endpoint ─────────────
print("=" * 60)
print("Step 1: Undeploy adult-test chain-scorer (free quota)")
print("=" * 60)

# Read endpoint ID from .env
adult_test_endpoint_id = None
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        if line.startswith("VERTEX_AI_ENDPOINT_ADULT_TEST="):
            adult_test_endpoint_id = line.strip().split("=", 1)[1]
            break

if not adult_test_endpoint_id:
    print("VERTEX_AI_ENDPOINT_ADULT_TEST not in .env — skipping undeploy")
else:
    try:
        ep = aiplatform.Endpoint(endpoint_name=adult_test_endpoint_id)
        deployed = ep.list_models()
        if deployed:
            for dm in deployed:
                print(f"  Undeploying model {dm.id} from endpoint {adult_test_endpoint_id}...")
                ep.undeploy(deployed_model_id=dm.id)
                print(f"  Done — quota freed")
        else:
            print(f"  Endpoint {adult_test_endpoint_id} has no deployed models (already free)")
    except Exception as e:
        print(f"  Undeploy error: {e}")
        print("  Continuing anyway...")

# ── Step 2: Also delete any leftover empty endpoint shells ───────────────────
print("\nStep 2: Delete empty endpoint shells from failed deploys")
EMPTY_IDS = [
    "4131050453263712256",   # adult-test outcome (empty)
    "2081349672856715264",   # german outcome (empty)
    "9114283440949166080",   # german outcome second attempt (empty)
]
for eid in EMPTY_IDS:
    try:
        ep = aiplatform.Endpoint(endpoint_name=eid)
        deployed = ep.list_models()
        if not deployed:
            ep.delete(force=True)
            print(f"  Deleted empty endpoint {eid}")
        else:
            print(f"  Endpoint {eid} has models — skipping delete")
    except Exception as e:
        print(f"  {eid}: {e} (may not exist — OK)")

# ── Step 3: Deploy german outcome-scorer ────────────────────────────────────
print("\nStep 3: Deploy auditra-outcome-scorer-german")
print("=" * 60)

display_name = "auditra-outcome-scorer-german"

models = aiplatform.Model.list(
    filter=f'display_name="{display_name}"',
    project=PROJECT_ID,
    location=REGION,
    order_by="create_time desc",
)
if not models:
    print(f"ERROR: No trained model '{display_name}'")
    exit(1)

model = models[0]
print(f"[deploy] Found: {display_name} ({model.resource_name})")

# Check for existing endpoint with a deployed model
endpoints = aiplatform.Endpoint.list(
    filter=f'display_name="{display_name}-endpoint"',
    project=PROJECT_ID,
    location=REGION,
)
eid = None
for ep in endpoints:
    deployed = ep.list_models()
    if deployed:
        eid = ep.resource_name.split("/")[-1]
        print(f"[deploy] Already deployed: endpoint {eid}")
        break

if not eid:
    endpoint = model.deploy(
        deployed_model_display_name=display_name,
        machine_type=MACHINE,
        min_replica_count=1,
        max_replica_count=1,
        traffic_split={"0": 100},
    )
    eid = endpoint.resource_name.split("/")[-1]
    print(f"[deploy] Done: {display_name} → endpoint {eid}")

# ── Write to .env ────────────────────────────────────────────────────────────
lines = []
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        lines = f.readlines()

key = "VERTEX_AI_OUTCOME_GERMAN"
new_lines = []
updated = False
for line in lines:
    if line.startswith(f"{key}="):
        new_lines.append(f"{key}={eid}\n")
        updated = True
    else:
        new_lines.append(line)
if not updated:
    new_lines.append(f"{key}={eid}\n")

with open(ENV_FILE, "w") as f:
    f.writelines(new_lines)

print(f"\n[env] {key}={eid} written to .env")
print("\n" + "=" * 60)
print("Done. Verify .env then restart server:")
print("  cat .env")
print("  pkill -f uvicorn")
print("  nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &")
print("=" * 60)
