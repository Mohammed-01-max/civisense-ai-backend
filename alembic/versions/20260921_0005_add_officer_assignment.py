"""Add officer role and deterministic report assignment fields.

Revision ID: 20260921_0005
Revises: 20260921_0004
Create Date: 2026-09-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260921_0005"
down_revision: Union[str, Sequence[str], None] = "20260921_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'officer'")
    op.add_column(
        "users",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "users",
        sa.Column("service_department", sa.String(), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column("assigned_officer_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_reports_assigned_officer_id_users",
        "reports",
        "users",
        ["assigned_officer_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_reports_assigned_officer_id_users",
        "reports",
        type_="foreignkey",
    )
    op.drop_column("reports", "assigned_officer_id")
    op.drop_column("users", "service_department")
    op.drop_column("users", "is_active")