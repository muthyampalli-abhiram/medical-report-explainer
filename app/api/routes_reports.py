"""FastAPI routes for lab reports management, explanations, and questions."""

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Path as PathParam, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.deps import get_db_path, get_llm
from app.api.uploads import validated_upload_file
from app.config import settings
from app.db.database import get_connection
from app.models.api import UploadResponse
from app.models.explanation import ReportExplanation
from app.models.lab import ExtractionResponse, LabReport, ResultFlag
from app.models.questions import DoctorQuestions
from app.services.explainer import FIXED_DISCLAIMER, explain_report
from app.services.llm_client import LLMClient
from app.services.ocr_extractor import (
    OCRUnavailableError,
    extract_report_from_image,
    extract_report_with_ocr_fallback,
)
from app.services.ocr_safeguards import apply_ocr_safeguards, filter_ocr_junk
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report
from app.services.questions import generate_questions
from app.services.repository import (
    DuplicateReportError,
    delete_report,
    get_report,
    get_trend,
    save_report,
)

router = APIRouter(tags=["reports"])


def _profile_exists(profile_id: int, db_path: Path) -> bool:
    """Check if profile exists in database."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT 1 FROM profiles WHERE id = ?;", (profile_id,))
        return cursor.fetchone() is not None
    finally:
        conn.close()


def _get_report_profile_id(report_id: str, db_path: Path) -> int | None:
    """Retrieve profile_id for a given report_id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT profile_id FROM reports WHERE id = ?;", (report_id,))
        row = cursor.fetchone()
        return row["profile_id"] if row else None
    finally:
        conn.close()


@router.post("/reports/upload", response_model=UploadResponse)
async def upload_report(
    file: UploadFile,
    profile_id: int = Form(1),
    db_path: Path = Depends(get_db_path),
):
    """Upload, extract, process, and save a digital lab report PDF or image."""
    if not _profile_exists(profile_id, db_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found.",
        )

    async with validated_upload_file(file, allow_images=True) as temp_path:
        is_pdf = temp_path.endswith(".pdf")

        try:
            if is_pdf:
                raw_response, extraction_method = extract_report_with_ocr_fallback(temp_path)
            else:
                raw_response = extract_report_from_image(temp_path)
                extraction_method = "ocr"
        except OCRUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OCR is not available on this server.",
            )

        # Never store or return the client filename
        raw_response.report.source_filename = None

        processed = process_report(raw_response)
        if extraction_method == "ocr":
            processed = filter_ocr_junk(processed)
            processed = apply_ocr_safeguards(processed)

        # For OCR extractions only: after filtering, if no remaining result has a canonical test name,
        # return HTTP 422 with the existing message and save nothing.
        if extraction_method == "ocr":
            has_canonical = any(r.test_name_canonical is not None for r in processed.report.results)
            if not has_canonical:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="We could not read any results from this file. Try a clearer, flat, well-lit photo of the full report.",
                )
        elif not processed.report.results:
            # Text-layer PDFs keep today's behavior exactly
            warn_detail = (
                "; ".join(processed.warnings)
                if processed.warnings
                else "No lab results could be extracted from this PDF."
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=warn_detail,
            )

        try:
            save_report(profile_id, processed, db_path)
        except DuplicateReportError as e:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "detail": "This report has already been uploaded.",
                    "existing_report_id": e.existing_report_id,
                },
            )

        abnormal_count = sum(
            1
            for r in processed.report.results
            if r.flag
            in (
                ResultFlag.HIGH,
                ResultFlag.LOW,
                ResultFlag.CRITICAL_LOW,
                ResultFlag.CRITICAL_HIGH,
            )
        )
        critical_count = sum(
            1
            for r in processed.report.results
            if r.flag in (ResultFlag.CRITICAL_LOW, ResultFlag.CRITICAL_HIGH)
        )
        low_confidence_count = sum(
            1 for r in processed.report.results if r.confidence < settings.OCR_VERIFY_BELOW
        )

        return UploadResponse(
            report_id=processed.report.report_id,
            report_date=processed.report.report_date,
            lab_name=processed.report.lab_name,
            result_count=len(processed.report.results),
            abnormal_count=abnormal_count,
            critical_count=critical_count,
            results=processed.report.results,
            warnings=processed.warnings,
            disclaimer=FIXED_DISCLAIMER,
            extraction_method=extraction_method,
            low_confidence_count=low_confidence_count,
        )


@router.get("/reports/{report_id}", response_model=LabReport)
def get_single_report(
    report_id: str = PathParam(...),
    db_path: Path = Depends(get_db_path),
):
    """Retrieve full details of an extracted lab report."""
    report = get_report(report_id, db_path)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )
    return report


@router.delete("/reports/{report_id}")
def delete_single_report(
    report_id: str = PathParam(...),
    db_path: Path = Depends(get_db_path),
):
    """Delete a report and cascade-delete its results."""
    success = delete_report(report_id, db_path)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )
    return {"deleted": True, "report_id": report_id}


@router.get("/reports/{report_id}/explanation", response_model=ReportExplanation)
def get_report_explanation(
    report_id: str = PathParam(...),
    lang: str = "en",
    db_path: Path = Depends(get_db_path),
    llm: LLMClient = Depends(get_llm),
):
    """Generate plain-language explanation for all biomarkers in a report."""
    if lang not in ("en", "hi", "te"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported language. Supported: en, hi, te.",
        )

    report = get_report(report_id, db_path)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )

    return explain_report(ExtractionResponse(report=report), llm=llm, language=lang)


@router.get("/reports/{report_id}/questions", response_model=DoctorQuestions)
def get_report_questions(
    report_id: str = PathParam(...),
    db_path: Path = Depends(get_db_path),
):
    """Generate deterministic, patient-oriented questions for their doctor."""
    report = get_report(report_id, db_path)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )

    profile_id = _get_report_profile_id(report_id, db_path)
    trends: dict[str, list] = {}
    if profile_id is not None:
        for res in report.results:
            if res.test_name_canonical and res.test_name_canonical not in trends:
                pts = get_trend(profile_id, res.test_name_canonical, db_path)
                if len(pts) >= 2:
                    trends[res.test_name_canonical] = pts

    return generate_questions(ExtractionResponse(report=report), trends=trends)
