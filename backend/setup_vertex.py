"""
Vertex AI one-time setup for Auditra — 4 datasets.

Step 1 — Run on the VM:
  pip install -r requirements.txt
  python setup_vertex.py

  Creates GCS bucket, uploads 4 datasets, launches 4 AutoML training jobs.
  Training runs in background (1-3 hours). Do NOT wait — move on.

Step 2 — After training completes:
  python deploy_vertex.py
"""
import io
import os
import sys

PROJECT_ID = "project-6bf0badc-9510-4a48-9e6"
REGION     = "us-central1"
BUCKET     = "auditra-ml-6bf0badc"
BUCKET_URI = f"gs://{BUCKET}"

# Minimum budget (1 node-hour) — keeps cost low, trains a valid model
BUDGET_MILLI_NODE_HOURS = 1000


def create_bucket():
    from google.cloud import storage
    client = storage.Client(project=PROJECT_ID)
    try:
        bucket = client.create_bucket(BUCKET, location=REGION)
        print(f"[bucket] Created: gs://{BUCKET}")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            bucket = client.bucket(BUCKET)
            print(f"[bucket] Already exists: gs://{BUCKET}")
        else:
            raise
    return bucket


def upload_dataset(bucket, df, name: str) -> str:
    csv_bytes = df.to_csv(index=False).encode()
    blob_path = f"datasets/{name}.csv"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(csv_bytes, content_type="text/csv")
    gcs_uri = f"{BUCKET_URI}/{blob_path}"
    print(f"[upload] {name}: {len(df)} rows → {gcs_uri}")
    return gcs_uri


def create_vertex_dataset(display_name: str, gcs_uri: str):
    from google.cloud import aiplatform
    ds = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=gcs_uri,
        project=PROJECT_ID,
        location=REGION,
    )
    print(f"[dataset] Created: {display_name} ({ds.resource_name})")
    return ds


def launch_automl_training(display_name: str, dataset, target_column: str) -> str:
    from google.cloud import aiplatform

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=display_name,
        optimization_prediction_type="classification",
        project=PROJECT_ID,
        location=REGION,
    )

    job.run(
        dataset=dataset,
        target_column=target_column,
        budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,
        model_display_name=display_name,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        sync=False,   # non-blocking — returns immediately
    )

    print(f"[train] Launched: {display_name}  target={target_column}")
    print(f"        Resource: {job.resource_name}")
    return job.resource_name


def load_adult_train_only() -> "pd.DataFrame":
    """Adult train split only (adult.data), NOT combined with test."""
    import pandas as pd
    import requests

    ADULT_COLS = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ]
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    local = os.path.join(data_dir, "adult_train.csv")

    if os.path.exists(local):
        raw = open(local, "rb").read()
    else:
        resp = requests.get(URL, timeout=30)
        resp.raise_for_status()
        os.makedirs(data_dir, exist_ok=True)
        with open(local, "wb") as f:
            f.write(resp.content)
        raw = resp.content

    df = pd.read_csv(io.BytesIO(raw), names=ADULT_COLS, skipinitialspace=True, na_values="?")
    df["income"] = df["income"].str.strip()
    df = df.drop(columns=["fnlwgt"], errors="ignore").dropna().reset_index(drop=True)
    return df


def load_adult_test_only() -> "pd.DataFrame":
    """Adult test split only (adult.test)."""
    import pandas as pd
    import requests

    ADULT_COLS = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ]
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    local = os.path.join(data_dir, "adult_test.csv")

    if os.path.exists(local):
        raw = open(local, "rb").read()
    else:
        resp = requests.get(URL, timeout=30)
        resp.raise_for_status()
        os.makedirs(data_dir, exist_ok=True)
        with open(local, "wb") as f:
            f.write(resp.content)
        raw = resp.content

    df = pd.read_csv(io.BytesIO(raw), names=ADULT_COLS, skipinitialspace=True,
                     na_values="?", skiprows=1)
    df["income"] = df["income"].str.strip().str.rstrip(".")
    df = df.drop(columns=["fnlwgt"], errors="ignore").dropna().reset_index(drop=True)
    return df


def main():
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("Auditra — Vertex AI Setup (4 datasets)")
    print(f"Project : {PROJECT_ID}")
    print(f"Region  : {REGION}")
    print(f"Bucket  : gs://{BUCKET}")
    print("=" * 60)

    # 1. Bucket
    print("\n[1/5] Creating GCS bucket...")
    bucket = create_bucket()

    # 2. Load all 4 datasets
    print("\n[2/5] Downloading datasets...")
    from app.services.data_loader import load_compas, load_german

    compas        = load_compas()
    adult_train   = load_adult_train_only()
    adult_test    = load_adult_test_only()
    german        = load_german()

    failed = [n for n, d in [("COMPAS", compas), ("Adult train", adult_train),
                               ("Adult test", adult_test), ("German", german)] if d is None]
    if failed:
        print(f"ERROR: Failed to download: {failed}")
        sys.exit(1)

    print(f"  COMPAS       : {len(compas)} rows")
    print(f"  Adult train  : {len(adult_train)} rows")
    print(f"  Adult test   : {len(adult_test)} rows")
    print(f"  German       : {len(german)} rows")

    # 3. Upload to GCS
    print("\n[3/5] Uploading to GCS...")
    compas_uri       = upload_dataset(bucket, compas,      "compas")
    adult_train_uri  = upload_dataset(bucket, adult_train, "adult_train")
    adult_test_uri   = upload_dataset(bucket, adult_test,  "adult_test")
    german_uri       = upload_dataset(bucket, german,      "german")

    # 4. Create Vertex AI TabularDatasets
    print("\n[4/5] Creating Vertex AI datasets...")
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION)

    ds_compas      = create_vertex_dataset("auditra-compas",      compas_uri)
    ds_adult_train = create_vertex_dataset("auditra-adult-train", adult_train_uri)
    ds_adult_test  = create_vertex_dataset("auditra-adult-test",  adult_test_uri)
    ds_german      = create_vertex_dataset("auditra-german",      german_uri)

    # 5. Launch 4 AutoML training jobs (non-blocking)
    # Model learns: can these features reconstruct the protected attribute?
    # Chain scorer sends only chain features; AutoML handles missing cols as null.
    print("\n[5/5] Launching AutoML training jobs (non-blocking)...")

    jobs = {
        "compas":      launch_automl_training("auditra-chain-scorer-compas",      ds_compas,      "race"),
        "adult_train": launch_automl_training("auditra-chain-scorer-adult-train", ds_adult_train, "sex"),
        "adult_test":  launch_automl_training("auditra-chain-scorer-adult-test",  ds_adult_test,  "sex"),
        "german":      launch_automl_training("auditra-chain-scorer-german",      ds_german,      "sex"),
    }

    with open("vertex_jobs.txt", "w") as f:
        for key, resource_name in jobs.items():
            f.write(f"{key}={resource_name}\n")

    print("\n" + "=" * 60)
    print("Setup complete. 4 training jobs launched.")
    print()
    print("Monitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training?project={PROJECT_ID}")
    print()
    print("When ALL 4 jobs show status = 'Succeeded', run:")
    print("  python deploy_vertex.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
