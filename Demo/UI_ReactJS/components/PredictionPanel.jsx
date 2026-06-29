function ProbabilityBar({ item }) {
  return (
    <div className="probability-bar">
      <div className="probability-bar-head">
        <span>{item.label}</span>
        <strong>{item.probability.toFixed(1)}%</strong>
      </div>
      <div className="probability-track">
        <div
          className="probability-fill"
          style={{ width: `${Math.max(item.probability, 2)}%` }}
        />
      </div>
    </div>
  );
}

export default function PredictionPanel({ status, result, error }) {
  const predictedWord = status === "success" && result ? result.predictedWord : " ";

  return (
    <section className="prediction-panel">
      <article className="prediction-result card">
        <h2>Results</h2>
        <div className="prediction-word-shell">
          <div className="prediction-word">{predictedWord}</div>
        </div>
      </article>

      <article className="prediction-details card">
        <div className="section-title">
          <img src="/elements/prediction_details_symbol.png" alt="" aria-hidden="true" />
          <h2>Prediction Details</h2>
        </div>

        <div className="prediction-summary">
          <div>
            <span>Predicted Word</span>
            <strong>{status === "success" && result ? result.predictedWord : "-"}</strong>
          </div>
          <div className="summary-right">
            <span>Confidence</span>
            <strong>{status === "success" && result ? `${result.confidence.toFixed(1)}%` : "-"}</strong>
          </div>
        </div>

        <div className="prediction-divider" />

        <div className="prediction-state">
          {status === "loading" ? (
            <p>Running prediction...</p>
          ) : null}

          {status === "error" ? (
            <p className="error-message">{error || "Prediction failed."}</p>
          ) : null}

          {status === "idle" ? (
            <p>Prediction results will appear here after you run the model.</p>
          ) : null}

          {status === "success" && result
            ? result.probabilities.map((item) => <ProbabilityBar key={item.label} item={item} />)
            : null}
        </div>
      </article>
    </section>
  );
}
