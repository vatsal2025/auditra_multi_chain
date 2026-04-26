import logging
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import audit, chat, demo, fix, report, upload

logger = logging.getLogger(__name__)

app = FastAPI(title="FairLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.on_event("startup")
async def _startup():
    # Run cache build in a daemon thread — avoids blocking the async event loop
    # (load_adult and LightGBM training are synchronous CPU/IO)
    import asyncio

    def _run():
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(demo.warm_adult_cache())
            loop.close()
        except Exception as e:
            logger.warning(f"Background demo cache thread failed: {e}")

    t = threading.Thread(target=_run, daemon=True, name="demo-cache-builder")
    t.start()


@app.get("/health")
def health():
    return {"status": "ok", "service": "FairLens API"}


# Serve React frontend if built dist exists
DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        index = os.path.join(DIST, "index.html")
        return FileResponse(index)
