"""initial schema — users, scans, cases, api_keys, contact_submissions

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-07 00:00:00.000000

Baseline migration that captures the current ORM-defined schema. Existing
deployments that bootstrapped via Base.metadata.create_all should run:

    alembic stamp 0001_initial

to mark this revision as already applied without re-creating tables.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── users ───
    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("language_pref", sa.String(length=10), server_default="en"),
        sa.Column("email_verified", sa.Boolean(), server_default=sa.text("0")),
        sa.Column("email_verify_code", sa.String(), nullable=True),
        sa.Column("email_verify_expires", sa.DateTime(), nullable=True),
        sa.Column("reset_token_hash", sa.String(), nullable=True),
        sa.Column("reset_token_expires", sa.DateTime(), nullable=True),
        sa.Column("totp_secret", sa.String(), nullable=True),
        sa.Column("totp_enabled", sa.Boolean(), server_default=sa.text("0")),
        sa.Column("backup_codes", sa.JSON(), nullable=True),
        sa.Column("oauth_provider", sa.String(length=20), nullable=True),
        sa.Column("notify_email_threats", sa.Boolean(), server_default=sa.text("1")),
        sa.Column("notify_email_reports", sa.Boolean(), server_default=sa.text("0")),
        sa.Column("notify_push_enabled", sa.Boolean(), server_default=sa.text("0")),
        sa.Column("notify_push_subscription", sa.JSON(), nullable=True),
        sa.Column("emergency_contact_name", sa.String(length=200), nullable=True),
        sa.Column("emergency_contact_phone", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ─── api_keys ───
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("key_hash", sa.String(), nullable=False),
        sa.Column("key_prefix", sa.String(length=12), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("1")),
        sa.Column("last_used", sa.DateTime(), nullable=True),
        sa.Column("request_count", sa.Integer(), server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"])

    # ─── scans ───
    op.create_table(
        "scans",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("file_type", sa.String(), nullable=False),
        sa.Column("verdict", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("forensic_data", sa.JSON(), nullable=True),
        sa.Column("raw_scores", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_scans_user_id", "scans", ["user_id"])

    # ─── cases ───
    op.create_table(
        "cases",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("scan_ids", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="open"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_cases_user_id", "cases", ["user_id"])
    op.create_index("ix_cases_status", "cases", ["status"])

    # ─── contact_submissions ───
    op.create_table(
        "contact_submissions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("subject", sa.String(length=500), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("contact_submissions")
    op.drop_index("ix_cases_status", table_name="cases")
    op.drop_index("ix_cases_user_id", table_name="cases")
    op.drop_table("cases")
    op.drop_index("ix_scans_user_id", table_name="scans")
    op.drop_table("scans")
    op.drop_index("ix_api_keys_key_hash", table_name="api_keys")
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
