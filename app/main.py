"""FastAPI application entrypoint with lifespan, routes, and exception handling."""

from contextlib import asynccontextmanager
import importlib.util
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes_prescriptions import router as prescriptions_router
from app.api.routes_profiles import router as profiles_router
from app.api.routes_reports import router as reports_router
from app.config import settings
from app.db.database import init_db

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _find_plotly_path() -> Path | None:
    """Locate package_data/plotly.min.js inside installed plotly package."""
    try:
        spec = importlib.util.find_spec("plotly")
        if spec and spec.origin:
            p = Path(spec.origin).resolve().parent / "package_data" / "plotly.min.js"
            if p.is_file():
                return p
    except Exception:
        pass
    return None


PLOTLY_PATH = _find_plotly_path()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown tasks."""
    init_db(settings.DB_PATH)
    yield


app = FastAPI(
    title="medical-report-explainer",
    description="Local-first medical report explainer API",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Set security headers on responses, exempting swagger docs and nocaching api data."""
    response = await call_next(request)
    path = request.url.path

    # Exempt Swagger/OpenAPI documentation paths from the strict CSP
    if not (
        path == "/docs"
        or path.startswith("/docs/")
        or path == "/redoc"
        or path.startswith("/redoc/")
        or path == "/openapi.json"
    ):
        # style-src allows inline styles only because Plotly injects them, and script-src must always stay 'self'. If the chart is blocked by the policy, relax style-src only, never script-src.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"

    # Set Cache-Control: no-store on every /api/* response (health data must not be cached by the browser)
    if path == "/api" or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"

    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler preventing information leaks on unhandled errors."""
    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Something went wrong while processing this request."},
    )


# Health check endpoint (kept unchanged)
@app.get("/health")
def health_check():
    """Health check endpoint confirming application status."""
    return {"status": "ok", "app": "medical-report-explainer"}


# Explicit route for vendor plotly.min.js before static mount to prevent shadowing
@app.get("/static/vendor/plotly.min.js")
def get_plotly_vendor():
    """Serve plotly.min.js directly from the installed plotly package."""
    if not PLOTLY_PATH or not PLOTLY_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="plotly.min.js not found in installed plotly package.",
        )
    return FileResponse(PLOTLY_PATH, media_type="application/javascript")


# Serve index.html at root
@app.get("/", response_class=FileResponse)
def get_index():
    """Serve the single-page application frontend."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="index.html not found.",
        )
    return FileResponse(index_file, media_type="text/html")


# Mount static assets directory
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Register API routers under /api prefix
app.include_router(profiles_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(prescriptions_router, prefix="/api")
