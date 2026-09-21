"""Add AI triage fields

Revision ID: 2b5d0990e561
Revises: 20260913_0001
Create Date: 2026-09-14 01:27:50.802787
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2b5d0990e561"
down_revision: Union[str, Sequence[str], None] = "20260913_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "reports",
        sa.Column("triage_category", sa.String(), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column("triage_confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column("triage_source", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("reports", "triage_source")
    op.drop_column("reports", "triage_confidence")
    op.drop_column("reports", "triage_category")
