"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload --app-dir backend
or from inside the backend/ directory:
    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as extraction_router
from app.core.config import settings
from app.schemas.extraction import HealthResponse
from app.services.model_loader import bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model + processor + OCR once, at startup, so the first request
    # isn't penalized with the cold-start cost.
    bundle.load()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"], response_model=HealthResponse)
def health():
    """Liveness probe — confirms the API process and model are up."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        model_loaded=bundle.is_loaded,
        device=bundle.device_str,
    )


app.include_router(extraction_router, prefix=settings.api_prefix)

# Serve saved annotated images as static files.
app.mount(
    settings.output_url_path,
    StaticFiles(directory=settings.output_dir),
    name="annotated",
)
