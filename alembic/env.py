"""Alembic environment for the async PostgreSQL application."""

import asyncio
import ssl
from urllib.parse import urlsplit

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from config import settings
from database import Base
import models  # noqa: F401  # Register all ORM tables on Base.metadata.


config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.split("?")[0])
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    context.configure(
        url=settings.DATABASE_URL.split("?")[0],
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    db_url = settings.DATABASE_URL.split("?")[0]
    hostname = (urlsplit(db_url).hostname or "").lower()
    connect_args = (
        {}
        if hostname in {"localhost", "127.0.0.1", "::1"}
        else {"ssl": ssl.create_default_context()}
    )
    connectable = create_async_engine(
        db_url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
