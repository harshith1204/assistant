from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PORT: int = 8080
    CRM_BASE_URL: str = "https://stage-api.simpo.ai/crm"
    CRM_BEARER_TOKEN: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(title="Research Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True, "service": "research-service-py"}


from .routers.research import router as research_router  # noqa: E402
app.include_router(research_router, prefix="/research", tags=["research"]) 
