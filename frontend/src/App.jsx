import { useState } from "react";

export default function App() {
  const [mode, setMode] = useState("document");
  const [file, setFile] = useState(null);
  const [text1, setText1] = useState("");
  const [text2, setText2] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const api = "http://127.0.0.1:8000";

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      let response;

      if (mode === "document") {
        if (!file) throw new Error("Please select a PDF, DOCX, or TXT file.");

        const data = new FormData();
        data.append("file", file);

        response = await fetch(`${api}/analyze`, {
          method: "POST",
          body: data,
        });
      } else {
        response = await fetch(`${api}/compare`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text1, text2 }),
        });
      }

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  function pct(value) {
    const n = Number(value ?? 0);
    return `${(n <= 1 ? n * 100 : n).toFixed(1)}%`;
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><span className="brand-icon">P</span> Plagiarism<span className="purple">AI</span></div>
        <span className="online"><i /> AI-powered text analysis</span>
      </header>

      <main className="container">
        <section className="hero">
          <span className="eyebrow">✦ INTELLIGENT SIMILARITY ANALYSIS</span>
          <h1>Originality starts<br />with <span className="purple">clarity.</span></h1>
          <p>Analyze documents and compare text using semantic similarity, text fingerprints, and source matching.</p>
          <div className="features"><span>✓ Semantic analysis</span><span>✓ Fingerprint matching</span><span>✓ Source detection</span></div>
        </section>

        <section className="panel">
          <p className="kicker">WORKSPACE</p>
          <h2>Analyze your content</h2>

          <div className="tabs">
            <button className={mode === "document" ? "tab selected" : "tab"} onClick={() => { setMode("document"); setResult(null); setError(""); }}>▤ Document analysis</button>
            <button className={mode === "compare" ? "tab selected" : "tab"} onClick={() => { setMode("compare"); setResult(null); setError(""); }}>⇄ Compare texts</button>
          </div>

          <form onSubmit={submit}>
            {mode === "document" ? (
              <div className="upload">
                <div className="upload-icon">↑</div>
                <h3>Upload your document</h3>
                <p>Choose a file to scan for potentially similar passages.</p>
                <label className="browse">Browse files
                  <input type="file" accept=".pdf,.docx,.txt" hidden onChange={e => { setFile(e.target.files[0] || null); setError(""); setResult(null); }} />
                </label>
                <small>PDF · DOCX · TXT</small>
                {file && <div className="file-name">Selected: {file.name} <button type="button" onClick={() => setFile(null)}>×</button></div>}
              </div>
            ) : (
              <div className="text-grid">
                <label>TEXT A<textarea required value={text1} onChange={e => setText1(e.target.value)} placeholder="Paste the first text here..." /></label>
                <label>TEXT B<textarea required value={text2} onChange={e => setText2(e.target.value)} placeholder="Paste the text to compare..." /></label>
              </div>
            )}

            <button className="primary" disabled={loading}>{loading ? "Analyzing..." : mode === "document" ? "Analyze document →" : "Compare texts →"}</button>
          </form>

          {error && <div className="error"><strong>Analysis failed</strong><p>{error}</p><small>Check that the backend is running at {api}.</small></div>}
        </section>

        {result && (
          <section className="panel results">
            <p className="kicker">ANALYSIS COMPLETE</p>
            <h2>Your results</h2>
            {mode === "compare" ? (
              <div className="metrics">
                <div><span>Semantic similarity</span><strong>{pct(result.semantic_similarity)}</strong></div>
                <div><span>Fingerprint similarity</span><strong>{pct(result.fingerprint_similarity)}</strong></div>
              </div>
            ) : (
              <>
                <div className="metrics">
                  <div><span>Characters analyzed</span><strong>{result.characters_analyzed ?? "—"}</strong></div>
                  <div><span>Chunks analyzed</span><strong>{result.chunks_analyzed ?? "—"}</strong></div>
                  <div><span>Sources checked</span><strong>{result.sources_checked ?? "—"}</strong></div>
                </div>
                <h3>Potentially similar passages</h3>
                {result.matches?.length ? result.matches.map((m, i) => (
                  <article className="match" key={i}>
                    <div className="match-head"><strong>{m.source_document || "Matched source"}</strong><span>{pct(m.fusion_score)} match score</span></div>
                    <div className="passages"><div><small>SUBMITTED PASSAGE</small><p>{m.submitted_passage || "—"}</p></div><div><small>SOURCE PASSAGE</small><p>{m.source_passage || "—"}</p></div></div>
                    <small>Semantic: {pct(m.semantic_similarity)} · Fingerprint: {pct(m.fingerprint_similarity)}</small>
                  </article>
                )) : <p>No matching passages were returned.</p>}
              </>
            )}
            <p className="disclaimer">Similarity scores are estimates, not proof of plagiarism. Review the passages and sources before drawing conclusions.</p>
          </section>
        )}

        <footer><span>PlagiarismAI</span><span>Understand similarity. Protect originality.</span></footer>
      </main>
    </div>
  );
}
