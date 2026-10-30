
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.detector import compare_texts
from app.extractor import extract_text
from app.retriever import retrieve_sources
from app.scoring import score_pair
from app.analyzer import analyze_text

app = FastAPI(
    title="AI Plagiarism Detector",
    description="Lexical and semantic document similarity analysis",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class CompareRequest(BaseModel):
    text1: str
    text2: str


@app.get("/")
def home():
    return {"message": "AI Plagiarism Detector API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compare")
def compare(request: CompareRequest):
    try:
        return score_pair(request.text1, request.text2)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload.txt").name
    extension = Path(filename).suffix.lower()

    if extension not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF, DOCX, or TXT file.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        submitted_text = extract_text(temp_path)

        candidates = retrieve_sources(submitted_text, top_k=5)
        matches = []

        for candidate in candidates:
            result = score_pair(
                submitted_text,
                candidate["source_text"],
            )

            matches.append({
                "source_document": candidate["source_document"],
                "bm25_score": candidate["bm25_score"],
                **result,
            })

        matches.sort(
            key=lambda item: item["fusion_score"],
            reverse=True,
        )

        result = analyze_text(submitted_text, top_k=5)

        return {
            "filename": filename,
            "characters_analyzed": len(submitted_text),
            **result,
            "note": (
                "Scores indicate similarity under the trained model. "
                "They are not proof of plagiarism."
            ),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
