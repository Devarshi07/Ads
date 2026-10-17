"""Alembic environment. Migrations are generated from theraflo_ads.infra.schema."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from theraflo_ads.infra.schema import metadata
from theraflo_ads.infra.settings import Settings

target_metadata = metadata


def run_migrations_offline() -> None:
    context.configure(url=Settings().database_url, target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(Settings().database_url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
