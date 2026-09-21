"""Add jurisdiction and deterministic routing fields.

Revision ID: 20260921_0004
Revises: cfecaccc83c3
Create Date: 2026-09-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260921_0004"
down_revision: Union[str, Sequence[str], None] = "cfecaccc83c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("jurisdiction", sa.String(), nullable=True))
    op.add_column("reports", sa.Column("jurisdiction", sa.String(), nullable=True))
    op.add_column("reports", sa.Column("service_department", sa.String(), nullable=True))
    op.add_column(
        "reports",
        sa.Column("routing_policy_version", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("reports", "routing_policy_version")
    op.drop_column("reports", "service_department")
    op.drop_column("reports", "jurisdiction")
    op.drop_column("users", "jurisdiction")