import re

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PLACEHOLDER_SECRET_PATTERN = re.compile(
    r"(?:^|[-_\s])(change|replace|your|example|default|template|placeholder|sample|dummy|test)(?:[-_\s]|$)",
    re.IGNORECASE,
)


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required secrets: never provide source-controlled fallbacks.
    DATABASE_URL: str
    SECRET_KEY: str

    # Browser origins are deliberately explicit because this API uses bearer tokens.
    CORS_ORIGINS: str

    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    # Password hashing scheme used by passlib's CryptContext. Common values:
    #  - pbkdf2_sha256 (no native deps)
    #  - bcrypt (preferred for many deployments, requires bcrypt wheel)
    HASHING_SCHEME: str = "pbkdf2_sha256"

    # API Configuration
    API_URL: str = "http://localhost:8000"

    # Analytics export
    ANALYTICS_EXCEL_PATH: str = "analytics/analytics.xlsx"

    # Upload safety limits. These defaults are intentionally conservative and
    # can be tuned through environment configuration without changing code.
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_IMAGE_WIDTH: int = 4096
    MAX_IMAGE_HEIGHT: int = 4096
    MAX_IMAGE_PIXELS: int = 16_000_000

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("DATABASE_URL must be configured")
        if not value.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
        return value

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        unsafe_values = {"change-this-to-a-long-random-secret", "your-secret-key", "secret"}
        normalized_value = value.strip()
        if (
            not normalized_value
            or len(normalized_value) < 32
            or normalized_value.lower() in unsafe_values
            or PLACEHOLDER_SECRET_PATTERN.search(normalized_value)
        ):
            raise ValueError("SECRET_KEY must be a non-placeholder value of at least 32 characters")
        return normalized_value

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        if not origins or "*" in origins:
            raise ValueError("CORS_ORIGINS must contain one or more explicit origins")
        return ",".join(origins)

    @property
    def cors_origins(self) -> list[str]:
        return self.CORS_ORIGINS.split(",")


settings = Settings()
