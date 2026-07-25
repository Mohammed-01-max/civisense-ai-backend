from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://neondb_owner:npg_4wtG6vxeXWzC@ep-autumn-leaf-apmsfltu-pooler.c-7.us-east-1.aws.neon.tech/neondb?ssl=require&channel_binding=require"

    # JWT
    SECRET_KEY: str = "change-this-to-a-long-random-secret"
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

    class Config:
        env_file = ".env"


settings = Settings()
