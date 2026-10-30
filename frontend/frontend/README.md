# AI Plagiarism Detector

An AI-assisted plagiarism detection web application that analyzes documents and identifies potentially similar passages from a reference corpus.

## Features

- Upload PDF, DOCX, and TXT documents.
- Extract text from uploaded files.
- Divide documents into smaller passages for analysis.
- Retrieve candidate source documents using BM25.
- Calculate character n-gram fingerprint similarity.
- Calculate semantic similarity using the pretrained MiniLM model.
- Combine similarity features using a trained Logistic Regression classifier.
- Display potential source documents and matching passages.

## Technology Stack

**Frontend:** React, Vite

**Backend:** Python, FastAPI

**Machine learning:** Sentence Transformers, scikit-learn, Logistic Regression

**Retrieval:** BM25

**Dataset:** PAN-PC-11 plagiarism corpus

## Model Design

The system uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` model to calculate semantic similarity. A separate Logistic Regression classifier combines semantic similarity, character-based similarity, and passage-length ratio to estimate the likelihood of a match.

The MiniLM model is pretrained; it is not fine-tuned by this project. The Logistic Regression classifier is trained separately using prepared positive and negative passage pairs.

## Running the Backend

From the `backend` directory, activate your virtual environment and run:

```bash
uvicorn app.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

## Running the Frontend

From the `frontend` directory, run:

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

## Important Limitation

A high similarity score is a signal for further review, not proof of plagiarism. Detection performance depends on the training data, source-corpus coverage, language, and type of rewriting.
