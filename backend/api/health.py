from fastapi import APIRouter
from fastapi.responses import ORJSONResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_class=ORJSONResponse)
async def health():
    return {"status": "ok", "service": "title4-callcenter-agent"}
