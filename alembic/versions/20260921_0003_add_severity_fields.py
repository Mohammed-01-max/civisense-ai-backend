"""Add severity fields to reports table.

Revision ID: 20260921_0003
Revises: 2b5d0990e561
Create Date: 2026-09-21

Adds three nullable columns to the reports table for the frozen
XGBoost-TF-IDF severity prediction (STEP 5 of CiviSense AI pipeline):

    severity            VARCHAR  — predicted class: LOW | MEDIUM | HIGH
    severity_confidence FLOAT    — max class probability from multi:softprob
    severity_source     VARCHAR  — model source identifier e.g. "xgboost-tfidf"

All columns are nullable so existing rows are unaffected and report
creation succeeds even when the severity service is unavailable.

Do NOT modify: 2b5d0990e561_add_ai_triage_fields.py
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260921_0003"
down_revision: Union[str, Sequence[str], None] = "2b5d0990e561"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "reports",
        sa.Column("severity", sa.String(), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column("severity_confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column("severity_source", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("reports", "severity_source")
    op.drop_column("reports", "severity_confidence")
    op.drop_column("reports", "severity")
