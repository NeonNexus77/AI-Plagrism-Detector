
import { useState } from "react";

export default function App() {
  const [mode, setMode] = useState("document");
  const [file, setFile] = useState(null);
  const [text1, setText1] = useState("");
  const [text2, setText2] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const API = "http://127.0.0.1:8000";

  async function compareTexts(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text1, text2 }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Text comparison failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  async function analyzeDocument(e) {
    e.preventDefault();

    if (!file) {
      setError("Please select a PDF, DOCX, or TXT file.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document analysis failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  function switchMode(nextMode) {
    setMode(nextMode);
    setResult(null);
    setError("");
  }

  function percentage(value) {
    const number = Number(value ?? 0);
    return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
  }

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">P</span>
          <span>Plagiarism<span className="brand-light">AI</span></span>
        </a>
        <span className="top-label">
          <span className="online-dot" />
          AI-Powered Text Analysis
        </span>
      </header>

      <main className="container">
        <section className="hero">
          <div className="eyebrow">
            <span>✦</span> INTELLIGENT SIMILARITY ANALYSIS
          </div>

          <h1>
            Originality starts
            <br />
            with <span>clarity.</span>
          </h1>

          <p className="hero-description">
            Analyze documents and compare text using semantic similarity,
            text fingerprints, and source matching.
          </p>

          <div className="feature-row">
            <span>✓ Semantic analysis</span>
            <span>✓ Fingerprint matching</span>
            <span>✓ Source detection</span>
          </div>
        </section>

        <section className="workspace">
          <div className="workspace-heading">
            <div>
              <p className="section-kicker">WORKSPACE</p>
              <h2>Analyze your content</h2>
            </div>
            <span className="secure-label">Private analysis</span>
          </div>

          <div className="tabs">
            <button
              type="button"
              className={mode === "document" ? "tab active" : "tab"}
              onClick={() => switchMode("document")}
            >
              <span>▤</span> Document analysis
            </button>
            <button
              type="button"
              className={mode === "compare" ? "tab active" : "tab"}
              onClick={() => switchMode("compare")}
            >
              <span>⇄</span> Compare texts
            </button>
          </div>

          {mode === "document" ? (
            <form onSubmit={analyzeDocument}>
              <div
                className="upload-area"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const droppedFile = e.dataTransfer.files[0];
                  if (droppedFile) {
                    setFile(droppedFile);
                    setError("");
                    setResult(null);
                  }
                }}
              >
                <div className="upload-icon">↑</div>
                <h3>Upload your document</h3>
                <p>Drag and drop a file here, or browse your computer.</p>

                <label className="browse-button">
                  Browse files
                  <input
                    type="file"
                    accept=".pdf,.docx,.txt"
                    onChange={(e) => {
                      setFile(e.target.files[0] || null);
                      setError("");
                      setResult(null);
                    }}
                    hidden
                  />
                </label>

                <div className="file-types">PDF · DOCX · TXT</div>
              </div>

              {file && (
                <div className="selected-file">
                  <span className="file-symbol">▤</span>
                  <div className="file-info">
                    <strong>{file.name}</strong>
                    <span>{(file.size / 1024).toFixed(1)} KB</span>
                  </div>
                  <button
                    type="button"
                    className="remove-file"
                    onClick={() => setFile(null)}
                    aria-label="Remove file"
                  >
                    ×
                  </button>
                </div>
              )}

              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "Analyzing document..." : "Analyze document  →"}
              </button>
            </form>
          ) : (
            <form onSubmit={compareTexts}>
              <div className="text-grid">
                <div className="text-field">
                  <label htmlFor="text1">TEXT A</label>
                  <textarea
                    id="text1"
                    value={text1}
                    onChange={(e) => setText1(e.target.value)}
                    placeholder="Paste your original or submitted text here..."
                    required
                    minLength={1}
                  />
                  <span>{text1.length} characters</span>
                </div>

                <div className="text-field">
                  <label htmlFor="text2">TEXT B</label>
                  <textarea
                    id="text2"
                    value={text2}
                    onChange={(e) => setText2(e.target.value)}
                    placeholder="Paste the text you want to compare it with..."
                    required
                    minLength={1}
                  />
                  <span>{text2.length} characters</span>
                </div>
              </div>

              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "Comparing texts..." : "Compare texts  →"}
              </button>
            </form>
          )}

          {error && (
            <div className="error-message" role="alert">
              <strong>Something went wrong</strong>
              <p>{error}</p>
              <span>Make sure the backend is running at {API}.</span>
            </div>
          )}
        </section>

        {result && (
          <section className="results">
            <div className="results-heading">
              <div>
                <p className="section-kicker">ANALYSIS COMPLETE</p>
                <h2>Your results</h2>
              </div>
              <span className="result-badge">✓ Complete</span>
            </div>

            {mode === "compare" ? (
              <div className="score-grid">
                <ScoreCard
                  title="Semantic similarity"
                  value={result.semantic_similarity}
                  description="Meaning-based similarity"
                  percentage={percentage}
                />
                <ScoreCard
                  title="Fingerprint similarity"
                  value={result.fingerprint_similarity}
                  description="Text pattern overlap"
                  percentage={percentage}
                />
              </div>
            ) : (
              <>
                <div className="stats-grid">
                  <StatCard
                    label="Characters analyzed"
                    value={result.characters_analyzed ?? "—"}
                  />
                  <StatCard
                    label="Chunks analyzed"
                    value={result.chunks_analyzed ?? "—"}
                  />
                  <StatCard
                    label="Sources checked"
                    value={result.sources_checked ?? "—"}
                  />
                </div>

                <h3 className="matches-title">Potentially similar passages</h3>

                {Array.isArray(result.matches) && result.matches.length > 0 ? (
                  <div className="match-list">
                    {result.matches.map((match, index) => (
                      <article className="match-card" key={index}>
                        <div className="match-header">
                          <strong>
                            {match.source_document || "Matched source"}
                          </strong>
                          <span className="match-score">
                            {percentage(match.fusion_score)} match score
                          </span>
                        </div>

                        <div className="passage-grid">
                          <div>
                            <span className="passage-label">
                              SUBMITTED PASSAGE
                            </span>
                            <p>{match.submitted_passage || "Not provided"}</p>
                          </div>
                          <div>
                            <span className="passage-label">SOURCE PASSAGE</span>
                            <p>{match.source_passage || "Not provided"}</p>
                          </div>
                        </div>

                        <div className="similarity-row">
                          <span>
                            Semantic:{" "}
                            {percentage(match.semantic_similarity)}
                          </span>
                          <span>
                            Fingerprint:{" "}
                            {percentage(match.fingerprint_similarity)}
                          </span>
                        </div>

                        {match.note && <p className="match-note">{match.note}</p>}
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="empty-results">
                    No matching passages were returned by the backend.
                  </div>
                )}
              </>
            )}

            <p className="disclaimer">
              Similarity scores are estimates, not proof of plagiarism.
              Review matched passages and their sources before drawing a
              conclusion.
            </p>
          </section>
        )}

        <footer>
          <span>PlagiarismAI</span>
          <span>Understand similarity. Protect originality.</span>
        </footer>
      </main>
    </div>
  );
}

function ScoreCard({ title, value, description, percentage }) {
  return (
    <div className="score-card">
      <p>{title}</p>
      <strong>{percentage(value)}</strong>
      <span>{description}</span>
      <div className="score-track">
        <div
          className="score-fill"
          style={{
            width: `${Math.min(
              100,
              Math.max(0, Number(value ?? 0) <= 1 ? Number(value ?? 0) * 100 : Number(value ?? 0))
            )}%`,
          }}
        />
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong>{Number(value).toLocaleString("en-IN") === "NaN" ? value : Number(value).toLocaleString("en-IN")}</strong>
    </div>
  );
}
