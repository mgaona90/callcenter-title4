import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.health import router as health_router
from api.chat import router as chat_router
from api.vapi_webhook import router as vapi_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Title IV Call Center Agent API starting")
    yield
    logger.info("Title IV Call Center Agent API stopped")


app = FastAPI(
    title="Title IV Call Center Agent",
    description=(
        "Voice AI agent for US federal financial aid (Title IV) and "
        "university enrollment verification. Handles ~2,500 call center seats."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router, prefix="/api")
app.include_router(vapi_router, prefix="/vapi")
