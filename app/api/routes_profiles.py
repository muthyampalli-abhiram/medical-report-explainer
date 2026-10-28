"""FastAPI routes for profiles and biomarker trends."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, status
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_db_path
from app.db.database import get_connection
from app.models.lab import LabResult
from app.models.trend import TrendPoint
from app.services.normalizer import canonicalize_name
from app.services.repository import (
    create_profile,
    delete_profile_data,
    get_latest_results,
    get_trend,
    list_profiles,
    list_reports,
)

router = APIRouter(tags=["profiles"])


class ProfileCreateRequest(BaseModel):
    """Payload to create a new profile. The label should be a nickname, not a legal name."""

    label: str = Field(
        ...,
        description="A nickname for the profile (e.g. 'Me', 'Mom'). Do not use a legal name.",
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        trimmed = v.strip()
        if len(trimmed) < 1 or len(trimmed) > 40:
            raise ValueError("Label must be between 1 and 40 characters.")
        return trimmed


def _profile_exists(profile_id: int, db_path: Path) -> bool:
    """Check if profile exists in database."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT 1 FROM profiles WHERE id = ?;", (profile_id,))
        return cursor.fetchone() is not None
    finally:
        conn.close()


@router.get("/profiles", response_model=list[dict[str, Any]])
def get_profiles(db_path: Path = Depends(get_db_path)):
    """List all user profiles."""
    return list_profiles(db_path)


@router.post(
    "/profiles",
    status_code=status.HTTP_201_CREATED,
    description="Create a new profile. Note: The label should be a nickname, not a legal name.",
)
def create_new_profile(
    payload: ProfileCreateRequest,
    db_path: Path = Depends(get_db_path),
):
    """Create a new profile with a unique nickname label."""
    try:
        profile_id = create_profile(payload.label, db_path)
        return {"id": profile_id, "label": payload.label}
    except ValueError as e:
        # Duplicate label or invalid
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.delete("/profiles/{profile_id}/data")
def delete_profile_reports(
    profile_id: int = PathParam(...),
    db_path: Path = Depends(get_db_path),
):
    """Delete all reports and associated results for a profile."""
    if not _profile_exists(profile_id, db_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")

    deleted_count = delete_profile_data(profile_id, db_path)
    return {"deleted_reports": deleted_count}


@router.get("/profiles/{profile_id}/reports", response_model=list[dict[str, Any]])
def get_profile_reports(
    profile_id: int = PathParam(...),
    db_path: Path = Depends(get_db_path),
):
    """List all reports stored for a specific profile."""
    if not _profile_exists(profile_id, db_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")

    return list_reports(profile_id, db_path)


@router.get("/profiles/{profile_id}/trends/{test_name}", response_model=list[TrendPoint])
def get_profile_trends(
    profile_id: int = PathParam(...),
    test_name: str = PathParam(...),
    db_path: Path = Depends(get_db_path),
):
    """Retrieve chronological trend points for a canonical biomarker."""
    if not _profile_exists(profile_id, db_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")

    canonical_name = canonicalize_name(test_name)
    if not canonical_name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown test name.",
        )

    return get_trend(profile_id, canonical_name, db_path)


@router.get("/profiles/{profile_id}/latest", response_model=dict[str, LabResult])
def get_profile_latest(
    profile_id: int = PathParam(...),
    db_path: Path = Depends(get_db_path),
):
    """Return the most recent numeric result for each canonical biomarker."""
    if not _profile_exists(profile_id, db_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")

    return get_latest_results(profile_id, db_path)
