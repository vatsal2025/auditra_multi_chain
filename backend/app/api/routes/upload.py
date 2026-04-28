import io
import uuid

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core import session_store
from app.models.schemas import ColumnInfo, UploadResponse
from app.services.graph_engine import detect_column_types

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset has no rows. Upload a CSV with data.")
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns.")

    session_id = str(uuid.uuid4())
    col_types = detect_column_types(df)
    session_store.set(session_id, "df", df)
    session_store.set(session_id, "col_types", col_types)
    session_store.set(session_id, "filename", file.filename)
    session_store.set(session_id, "chat_history", [])
    session_store.set(session_id, "fixes_applied", [])

    columns = [
        ColumnInfo(
            name=col,
            dtype=col_types[col],
            unique_count=int(df[col].nunique()),
            null_pct=round(float(df[col].isnull().mean()), 4),
        )
        for col in df.columns
    ]

    return UploadResponse(
        session_id=session_id,
        columns=columns,
        row_count=len(df),
    )
