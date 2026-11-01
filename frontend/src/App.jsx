
import { useEffect, useState } from 'react'
import './App.css'

const navigation = [
  { id: 'classify', label: 'Classify', icon: '◉' },
  { id: 'dataset', label: 'Dataset', icon: '▤' },
  { id: 'analytics', label: 'Analytics', icon: '▥' },
  { id: 'review', label: 'Review', icon: '☷' },
]

const examples = [
  'The company reported a 12% increase in quarterly revenue compared to last year.',
  'Investors should diversify their portfolios to reduce exposure to market volatility.',
  'It remains unclear whether the stock price will rise or fall next quarter.',
]

function App() {
  const [page, setPage] = useState('classify')
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState(() => {
  try {
    return JSON.parse(localStorage.getItem('finguard-history') || '[]')
  } catch {
    return []
  }
})

useEffect(() => {
  localStorage.setItem('finguard-history', JSON.stringify(history))
}, [history])

  async function classifyText(event) {
    event.preventDefault()
    if (!text.trim()) return

    setLoading(true)
    setResult(null)
    setError('')

    try {
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const response = await fetch(`${API_BASE_URL}/api/classify`,  {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Classification failed.')
      }

      setResult(data)
      setHistory((previous) => [
        {
          id: Date.now(),
          text: text.trim(),
          label: data.label,
          explanation: data.explanation,
          model: data.model,
        },
        ...previous,
      ])
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to FastAPI. Check that the backend is running on port 8000.'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  function chooseExample(example) {
    setText(example)
    setResult(null)
    setError('')
    setPage('classify')
  }

  const counts = history.reduce(
    (totals, item) => {
      totals[item.label] = (totals[item.label] || 0) + 1
      return totals
    },
    {}
  )

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setPage('classify')}>
          <span className="brand-mark">F</span>
          <span>
            <strong>FinGuard</strong>
            <small>Financial intelligence</small>
          </span>
        </button>

        <div className="sidebar-caption">WORKSPACE</div>
        <nav className="navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? 'active' : ''}`}
              onClick={() => setPage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
              {item.id === 'dataset' && history.length > 0 && (
                <span className="nav-count">{history.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <span className="connection-dot" />
          <span>Classification service</span>
          <strong>Groq</strong>
          <small>Results are generated on request</small>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">FinGuard <span>/</span> {navigation.find((item) => item.id === page)?.label || 'Classify'}</div>
          <div className="topbar-status"><span className="connection-dot" /> AI workspace</div>
        </header>

        <div className="page-content">
          {page === 'classify' && (
            <>
              <section className="page-heading">
                <div className="eyebrow">FINANCIAL LANGUAGE INTELLIGENCE</div>
                <h1>Understand the meaning<br />behind financial language.</h1>
                <p>Distinguish factual information, financial advice, and statements where the outcome remains uncertain.</p>
              </section>

              <section className="classify-grid">
                <form className="panel analyzer" onSubmit={classifyText}>
                  <div className="panel-heading">
                    <div>
                      <h2>Analyze a statement</h2>
                      <p>Paste financial text below to classify it.</p>
                    </div>
                    <span className="step-badge">01 / ANALYZE</span>
                  </div>

                  <label htmlFor="financial-text">FINANCIAL TEXT</label>
                  <textarea
                    id="financial-text"
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    placeholder="Paste a financial statement, market commentary, or AI-generated response..."
                    rows={8}
                    maxLength={5000}
                  />

                  <div className="input-footer">
                    <span>{text.length.toLocaleString()} / 5,000 characters</span>
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => {
                        setText('')
                        setResult(null)
                        setError('')
                      }}
                    >
                      Clear text
                    </button>
                  </div>

                  <button className="primary-button" type="submit" disabled={loading || !text.trim()}>
                    {loading ? 'Analyzing statement…' : 'Classify statement'}
                    {!loading && <span>↗</span>}
                  </button>

                  {error && <p className="error-message">{error}</p>}

                  <div className="examples">
                    <div className="examples-title">TRY AN EXAMPLE</div>
                    {examples.map((example, index) => (
                      <button
                        type="button"
                        className="example-button"
                        key={example}
                        onClick={() => chooseExample(example)}
                      >
                        <span>0{index + 1}</span>
                        {example}
                      </button>
                    ))}
                  </div>
                </form>

                <section className="panel result-panel" aria-live="polite">
                  <div className="panel-heading">
                    <div>
                      <h2>Classification result</h2>
                      <p>Analysis and model explanation</p>
                    </div>
                    <span className="result-symbol">✳</span>
                  </div>

                  {result ? (
                    <div className="result-content">
                      <div className="result-label">PREDICTED CATEGORY</div>
                      <div className={`category category-${String(result.label || '').toLowerCase()}`}>
                        <span className="category-dot" />
                        {result.label}
                      </div>
                      <div className="result-divider" />
                      <h3>Why this label?</h3>
                      <p className="explanation">{result.explanation || 'No explanation was provided.'}</p>
                      {result.model && (
                        <div className="model-card">
                          <span>MODEL</span>
                          <strong>{result.model}</strong>
                        </div>
                      )}
                      <p className="review-note">AI-generated classification. Review the result before relying on it.</p>
                    </div>
                  ) : (
                    <div className="empty-result">
                      <div className="empty-illustration"><span>F</span><i>✳</i></div>
                      <h3>Your analysis starts here</h3>
                      <p>Submit a statement to see its predicted category and the reasoning behind the result.</p>
                      <div className="label-guide">
                        <div><span className="guide-dot informational" /><span><strong>Informational</strong><small>Facts without a recommendation</small></span></div>
                        <div><span className="guide-dot advisory" /><span><strong>Advisory</strong><small>Suggests a financial action</small></span></div>
                        <div><span className="guide-dot uncertain" /><span><strong>Uncertain</strong><small>An unresolved financial outcome</small></span></div>
                      </div>
                    </div>
                  )}
                </section>
              </section>

              <section className="bottom-note">
                <span>i</span>
                <p>FinGuard identifies linguistic categories, not whether a financial decision is suitable for a particular person.</p>
              </section>
            </>
          )}

          {page === 'dataset' && (
            <>
              <section className="page-heading compact-heading">
                <div className="eyebrow">YOUR WORKSPACE</div>
                <h1>Classification history</h1>
                <p>Review statements classified during this browser session.</p>
              </section>
              <section className="panel data-panel">
                <div className="panel-heading">
                  <div><h2>Recent statements</h2><p>{history.length} classification{history.length === 1 ? '' : 's'} in this session</p></div>
                  <button className="text-button" onClick={() => setHistory([])} disabled={!history.length}>Clear history</button>
                </div>
                {history.length ? (
                  <div className="history-list">
                    {history.map((item) => (
                      <button className="history-row" key={item.id} onClick={() => {
                        setText(item.text)
                        setResult(item)
                        setError('')
                        setPage('classify')
                      }}>
                        <span className={`history-label category-${String(item.label || '').toLowerCase()}`}>{item.label}</span>
                        <span className="history-text">{item.text}</span>
                        <span className="history-arrow">↗</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state"><span>▤</span><h3>No statements yet</h3><p>Classify a financial statement and it will appear here for this session.</p><button className="secondary-button" onClick={() => setPage('classify')}>Go to Classify</button></div>
                )}
              </section>
            </>
          )}

          
{page === 'analytics' && (
  <>
    <section className="page-heading compact-heading">
      <div className="eyebrow">PERFORMANCE OVERVIEW</div>
      <h1>Analytics overview</h1>
      <p>Understand the distribution of your financial text classifications.</p>
    </section>

    <section className="stats-grid">
      <div className="panel stat-card">
        <span>Total classifications</span>
        <strong>{history.length}</strong>
        <small>Saved in this browser</small>
      </div>
      <div className="panel stat-card">
        <span>Informational</span>
        <strong>{counts.Informational || 0}</strong>
        <small>Factual statements</small>
      </div>
      <div className="panel stat-card">
        <span>Advisory</span>
        <strong>{counts.Advisory || 0}</strong>
        <small>Action-oriented statements</small>
      </div>
      <div className="panel stat-card">
        <span>Uncertain</span>
        <strong>{counts.Uncertain || 0}</strong>
        <small>Unresolved outcomes</small>
      </div>
    </section>

    <section className="analytics-layout">
      <div className="panel analytics-panel">
        <div className="panel-heading">
          <div>
            <h2>Category distribution</h2>
            <p>Breakdown of your saved classifications</p>
          </div>
        </div>

        {history.length > 0 ? (
          <div className="distribution">
            {['Informational', 'Advisory', 'Uncertain'].map((label) => {
              const count = counts[label] || 0
              const percent = Math.round((count / history.length) * 100)

              return (
                <div className="distribution-row" key={label}>
                  <div className="distribution-label">
                    <span className={`guide-dot ${label.toLowerCase()}`} />
                    {label}
                    <strong>{count}</strong>
                  </div>
                  <div className="distribution-track">
                    <span
                      className={`distribution-fill ${label.toLowerCase()}`}
                      style={{ width: `${percent}%` }}
                    />
                  </div>
                  <small>{percent}%</small>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="empty-state">
            <span>▥</span>
            <h3>No data available yet</h3>
            <p>Classify a few statements to populate this chart.</p>
            <button
              className="secondary-button"
              onClick={() => setPage('classify')}
            >
              Classify a statement
            </button>
          </div>
        )}
      </div>

      <div className="panel analytics-panel activity-panel">
        <div className="panel-heading">
          <div>
            <h2>Recent activity</h2>
            <p>Your latest classification results</p>
          </div>
          <button
            className="text-button"
            onClick={() => setPage('dataset')}
          >
            View all →
          </button>
        </div>

        {history.length > 0 ? (
          <div className="activity-list">
            {history.slice(0, 5).map((item) => (
              <button
                className="activity-item"
                key={item.id}
                onClick={() => {
                  setText(item.text)
                  setResult(item)
                  setPage('classify')
                }}
              >
                <span
                  className={`guide-dot ${String(item.label).toLowerCase()}`}
                />
                <span className="activity-copy">
                  <strong>{item.label}</strong>
                  <small>{item.text}</small>
                </span>
                <span className="history-arrow">↗</span>
              </button>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <span>◷</span>
            <h3>Nothing to display</h3>
            <p>Your recent results will appear here.</p>
          </div>
        )}
      </div>
    </section>

    <p className="analytics-disclaimer">
      Analytics reflect saved results in this browser only. These counts
      do not represent model accuracy or benchmark performance.
    </p>
  </>
)}


          {page === 'review' && (
            <>
              <section className="page-heading compact-heading">
                <div className="eyebrow">QUALITY & INTERPRETATION</div>
                <h1>Review classifications</h1>
                <p>Inspect recent outputs and decide which statements need closer human review.</p>
              </section>
              <section className="panel data-panel">
                <div className="panel-heading"><div><h2>Statements to inspect</h2><p>Recent outputs are listed for manual review.</p></div></div>
                {history.length ? (
                  <div className="history-list">
                    {history.map((item) => (
                      <div className="review-row" key={item.id}>
                        <div className="review-row-top"><span className={`history-label category-${String(item.label || '').toLowerCase()}`}>{item.label}</span><button className="text-button" onClick={() => {
                          setText(item.text)
                          setResult(item)
                          setPage('classify')
                        }}>Inspect result ↗</button></div>
                        <p>{item.text}</p><small>{item.explanation}</small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state"><span>☷</span><h3>Nothing to review yet</h3><p>Your classification outputs will be available here after you analyze statements.</p><button className="secondary-button" onClick={() => setPage('classify')}>Start classifying</button></div>
                )}
              </section>
            </>
          )}
        </div>

        <footer className="app-footer">
          <span>FinGuard <span className="footer-separator">/</span> Financial Language Intelligence</span>
          <span>AI outputs may require human review.</span>
        </footer>
      </main>
    </div>
  )
}

export default App
