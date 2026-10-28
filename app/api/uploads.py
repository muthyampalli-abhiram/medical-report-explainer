"""Upload validation and temporary file management."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import tempfile
from typing import AsyncGenerator

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import settings

# Dedicated temporary directory for uploaded files
UPLOAD_TEMP_DIR = Path(tempfile.gettempdir()) / "medical_report_explainer_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)


async def read_validated_upload(file: UploadFile, allow_images: bool = False) -> str:
    """Read and validate an uploaded file, returning the temporary file path.

    Validations:
      - PDF: starts with b"%PDF-"
      - JPEG: starts with b"\xff\xd8\xff" (if allow_images=True)
      - PNG: starts with b"\x89PNG\r\n\x1a\n" (if allow_images=True)
      - HEIC/HEIF: bytes 4-12 contain "ftyp" + heic/heix/mif1/hevc -> HTTP 415
      - Size limit: MAX_UPLOAD_MB -> HTTP 413 "File is too large."
      - Empty file: HTTP 400 "The uploaded file is empty."
      - Images verified with Pillow Image.open(...).verify() and pixel limit
    """
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    first_chunk = await file.read(64 * 1024)
    if not first_chunk:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # 1. HEIC/HEIF check
    if len(first_chunk) >= 12:
        ftyp_idx = first_chunk[:16].find(b"ftyp")
        if ftyp_idx != -1:
            tail = first_chunk[ftyp_idx + 4 : ftyp_idx + 12].lower()
            if any(b in tail for b in [b"heic", b"heix", b"mif1", b"hevc"]):
                raise HTTPException(
                    status_code=415,
                    detail="HEIC photos are not supported. Please save the photo as JPG or PNG.",
                )

    # 2. Type detection via magic bytes
    if first_chunk.startswith(b"%PDF-"):
        suffix = ".pdf"
    elif allow_images and first_chunk.startswith(b"\xff\xd8\xff"):
        suffix = ".jpg"
    elif allow_images and first_chunk.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = ".png"
    else:
        if allow_images:
            raise HTTPException(
                status_code=415,
                detail="Supported formats: PDF, JPG, PNG.",
            )
        else:
            raise HTTPException(
                status_code=415,
                detail="Only PDF files are supported right now.",
            )

    # Create random temporary file in dedicated directory
    fd, temp_path = tempfile.mkstemp(dir=UPLOAD_TEMP_DIR, prefix="upload_", suffix=suffix)
    temp_file = os.fdopen(fd, "wb")

    try:
        total_bytes = len(first_chunk)
        if total_bytes > max_bytes:
            raise HTTPException(status_code=413, detail="File is too large.")

        temp_file.write(first_chunk)

        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(status_code=413, detail="File is too large.")
            temp_file.write(chunk)

        temp_file.close()

        # Validate images with Pillow
        if suffix in (".jpg", ".png"):
            Image.MAX_IMAGE_PIXELS = settings.OCR_MAX_IMAGE_PIXELS
            try:
                with Image.open(temp_path) as img:
                    if img.width * img.height > settings.OCR_MAX_IMAGE_PIXELS:
                        raise HTTPException(status_code=413, detail="Image is too large.")
                    img.verify()
            except Image.DecompressionBombError:
                raise HTTPException(status_code=413, detail="Image is too large.")
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=415, detail="The image could not be read.")

        return temp_path

    except Exception:
        try:
            temp_file.close()
        except Exception:
            pass
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


async def read_validated_pdf(file: UploadFile) -> str:
    """Read and validate an uploaded PDF file, returning the temporary file path."""
    return await read_validated_upload(file, allow_images=False)


@asynccontextmanager
async def validated_upload_file(
    file: UploadFile, allow_images: bool = False
) -> AsyncGenerator[str, None]:
    """Context manager yielding temporary validated file path and guaranteeing deletion."""
    temp_path = await read_validated_upload(file, allow_images=allow_images)
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@asynccontextmanager
async def validated_pdf_file(file: UploadFile) -> AsyncGenerator[str, None]:
    """Context manager yielding temporary PDF file path and guaranteeing deletion."""
    async with validated_upload_file(file, allow_images=False) as temp_path:
        yield temp_path
