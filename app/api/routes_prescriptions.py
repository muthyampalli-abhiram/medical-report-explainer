"""FastAPI routes for prescription parsing and summarization."""

from fastapi import APIRouter, UploadFile

from app.api.uploads import validated_upload_file
from app.models.prescription import PrescriptionSummary
from app.services.prescription import parse_prescription_pdf

router = APIRouter(tags=["prescriptions"])


@router.post("/prescriptions/parse", response_model=PrescriptionSummary)
async def parse_prescription(file: UploadFile):
    """Parse and summarize an uploaded prescription PDF without storing data."""
    async with validated_upload_file(file, allow_images=False) as temp_pdf_path:
        summary = parse_prescription_pdf(temp_pdf_path)
        return summary
