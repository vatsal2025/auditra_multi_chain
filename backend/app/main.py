from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import audit, chat, demo, fix, report, upload

app = FastAPI(title="FairLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(demo.router, prefix="/api")
app.include_router(audit.router, prefix="/api")
app.include_router(fix.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(report.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "service": "FairLens API"}
