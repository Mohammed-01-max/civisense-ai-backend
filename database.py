import ssl
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import settings

# Strip any ?ssl=... query param from the URL and handle SSL via connect_args instead.
# This avoids asyncpg receiving unknown keyword arguments like 'sslmode' or 'channel_binding'.
db_url = settings.DATABASE_URL.split("?")[0]

ssl_ctx = ssl.create_default_context()

engine = create_async_engine(
    db_url,
    echo=False,
    connect_args={"ssl": ssl_ctx},
    pool_pre_ping=True,
    pool_recycle=1800,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()