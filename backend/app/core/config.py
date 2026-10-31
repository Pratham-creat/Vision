from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VISION"
    max_upload_size: int = 15 * 1024 * 1024
    storage_dir: Path = Path("storage")
    database_url: str = "sqlite:///./storage/vision.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_prefix="VISION_", env_file=".env")

    @property
    def cors_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)
