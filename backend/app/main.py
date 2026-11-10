from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.config.settings import settings
logger = settings.get_logger()

from app.routers.clinical_decision_router import router as clinical_decision_router
from app.dependencies import (
    initialize_dependencies,
    shutdown_dependencies,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifecycle.

    Startup:
        Initialize all long-lived application dependencies.

    Shutdown:
        Release/close application resources.
    """

    logger.info("Application startup initiated")

    try:
        # ---------------------------------------------------------
        # Initialize application dependencies
        # ---------------------------------------------------------
        await initialize_dependencies()

        logger.info(
            "Application dependencies initialized successfully"
        )

        yield

    except Exception:
        logger.exception(
            "Application startup failed"
        )
        raise

    finally:
        # ---------------------------------------------------------
        # Shutdown dependencies
        # ---------------------------------------------------------
        logger.info("Application shutdown initiated")

        try:
            await shutdown_dependencies()

            logger.info(
                "Application dependencies shut down successfully"
            )

        except Exception:
            logger.exception(
                "Error while shutting down application dependencies"
            )


app = FastAPI(
    title="Clinical Decision Support & Diagnosis API",
    version="1.0.0",
    description=(
        "Agentic Clinical Decision Support System "
        "using LangGraph and LangChain."
    ),
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(
    clinical_decision_router
)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.get("/ready")
async def ready():
    return {
        "status": "ready"
    }


# ------------------------------------------------------------------
# Global exception handler
# ------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled application exception | "
        f"path={request.url.path}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        },
    )