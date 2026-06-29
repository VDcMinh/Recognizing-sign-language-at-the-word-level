import { useEffect, useMemo, useRef, useState } from "react";
import { fetchSettings, predictVideo } from "./api";

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function createUploadItem(file) {
  return {
    id: `${file.name}:${file.size}:${file.lastModified}`,
    file,
    name: file.name,
    size: file.size,
    objectUrl: URL.createObjectURL(file)
  };
}

function Header({ activeSubset, poseReady }) {
  return (
    <header className="page-header">
      <span className="eyebrow">React Demo UI</span>
      <h1>Word-level sign language inference from one uploaded video</h1>
      <p>
        The UI keeps the visual direction from the temporary Demo folder, but now
        talks to the real local backend. The active subset is fixed in backend code,
        and every branch prediction uses the corresponding `best.pt` for that subset.
      </p>
      <div className="header-meta">
        <span className="meta-chip">Active subset: {activeSubset || "loading..."}</span>
        <span className={`meta-chip ${poseReady ? "meta-chip-success" : "meta-chip-error"}`}>
          RTMW pose backend: {poseReady ? "ready" : "not ready"}
        </span>
      </div>
    </header>
  );
}

function ActionBar({
  branches,
  selectedBranch,
  onBranchChange,
  onUpload,
  onPredict,
  canPredict,
  loading
}) {
  const inputRef = useRef(null);

  return (
    <section className="card action-bar">
      <div className="action-group">
        <button
          type="button"
          className="action-button action-button-secondary"
          onClick={() => inputRef.current?.click()}
        >
          Upload video
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.avi,.mkv,.webm,.m4v"
          className="hidden-input"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onUpload(file);
            }
            event.target.value = "";
          }}
        />
        <div className="model-select">
          <label htmlFor="branch-select" className="select-label">
            Branch
          </label>
          <select
            id="branch-select"
            value={selectedBranch}
            onChange={(event) => onBranchChange(event.target.value)}
          >
            {branches.map((branch) => (
              <option key={branch.id} value={branch.id}>
                {branch.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <button
        type="button"
        className="action-button action-button-primary"
        disabled={!canPredict || loading}
        onClick={onPredict}
      >
        {loading ? "Running..." : "Predict"}
      </button>
    </section>
  );
}

function VideoPlayer({ video }) {
  return (
    <section className="card video-player">
      <div className="video-stage">
        {video ? (
          <video className="video-element" src={video.objectUrl} controls playsInline />
        ) : (
          <div className="video-empty-state">
            <div>
              <h2>No video selected</h2>
              <p>Upload one video to preview it here, then choose a branch and run prediction.</p>
            </div>
          </div>
        )}
      </div>
      {video ? (
        <div className="video-controls">
          <div className="player-status">
            {video.name} - {(video.size / (1024 * 1024)).toFixed(2)} MB
          </div>
        </div>
      ) : null}
    </section>
  );
}

function RecentVideos({ videos, selectedId, onSelect, onRemove }) {
  return (
    <section className="card recent-videos">
      <div className="section-title">
        <h2>Recent uploads</h2>
      </div>
      <div className="recent-videos-track">
        {videos.length === 0 ? (
          <div className="empty-recent">Uploaded videos will appear here.</div>
        ) : (
          videos.map((video) => (
            <article
              key={video.id}
              role="button"
              tabIndex={0}
              className={`video-card ${selectedId === video.id ? "is-selected" : ""}`}
              onClick={() => onSelect(video.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(video.id);
                }
              }}
            >
              <div className="video-card-thumb">
                <video src={video.objectUrl} muted />
                <button
                  type="button"
                  className="delete-chip"
                  onClick={(event) => {
                    event.stopPropagation();
                    onRemove(video.id);
                  }}
                >
                  x
                </button>
              </div>
              <div className="video-card-body">
                <div className="video-card-heading">
                  <h3>{video.name}</h3>
                </div>
                <p>{(video.size / (1024 * 1024)).toFixed(2)} MB</p>
                <span className="source-chip">local upload</span>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}

function PredictionPanel({ loading, error, result, backendInfo }) {
  const topPrediction = result?.top_prediction || null;

  return (
    <section className="prediction-panel">
      <div className="card prediction-result">
        <h2>Prediction result</h2>
        {loading ? (
          <div className="prediction-state">
            <p>The backend is preprocessing the video, extracting pose, building tensors, and running the selected model.</p>
          </div>
        ) : null}
        {!loading && error ? (
          <div className="prediction-state">
            <p className="error-message">{error}</p>
          </div>
        ) : null}
        {!loading && !error && !result ? (
          <div className="prediction-state">
            <p>Pick one branch and run prediction to inspect the top-k output from the active subset.</p>
          </div>
        ) : null}
        {!loading && !error && result ? (
          <>
            <div className="prediction-word-shell">
              <div className="prediction-word">{topPrediction?.gloss || "N/A"}</div>
            </div>
            <div className="prediction-summary">
              <div>
                <span>Confidence</span>
                <strong>{formatPercent(topPrediction?.confidence)}</strong>
              </div>
              <div className="summary-right">
                <span>Branch</span>
                <strong>{result.branch_label}</strong>
              </div>
            </div>
            <div className="prediction-divider" />
            {result.predictions.map((item) => (
              <div className="probability-bar" key={`${item.class_id}:${item.rank}`}>
                <div className="probability-bar-head">
                  <span>
                    #{item.rank} {item.gloss}
                  </span>
                  <strong>{formatPercent(item.confidence)}</strong>
                </div>
                <div className="probability-track">
                  <div
                    className="probability-fill"
                    style={{ width: `${Math.max(2, item.confidence * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </>
        ) : null}
      </div>

      <div className="card prediction-details">
        <h2>Backend details</h2>
        <div className="details-grid">
          <div className="detail-item">
            <span>Active subset</span>
            <strong>{backendInfo?.active_subset || "N/A"}</strong>
          </div>
          <div className="detail-item">
            <span>Pose backend</span>
            <strong>{backendInfo?.pose_backend_ready ? "ready" : "not ready"}</strong>
          </div>
          <div className="detail-item">
            <span>Job id</span>
            <strong>{result?.job_id || "-"}</strong>
          </div>
          <div className="detail-item">
            <span>Frames</span>
            <strong>{result?.processing?.num_frames_standardized ?? "-"}</strong>
          </div>
          <div className="detail-item">
            <span>Pose frames</span>
            <strong>{result?.processing?.num_frames_pose ?? "-"}</strong>
          </div>
          <div className="detail-item">
            <span>Fusion gate mean</span>
            <strong>{result?.extra?.gate_mean?.toFixed?.(4) ?? "-"}</strong>
          </div>
        </div>
        {backendInfo?.pose_backend_error ? (
          <p className="backend-warning">{backendInfo.pose_backend_error}</p>
        ) : null}
        {result ? (
          <div className="notes-block">
            <h3>Pipeline notes</h3>
            <ul className="notes-list">
              {Object.entries(result.notes || {}).map(([key, value]) => (
                <li key={key}>
                  <strong>{key}:</strong> {value || "none"}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default function App() {
  const [backendInfo, setBackendInfo] = useState(null);
  const [videos, setVideos] = useState([]);
  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [selectedBranch, setSelectedBranch] = useState("skeleton");
  const [loadingSettings, setLoadingSettings] = useState(true);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const revokeQueueRef = useRef([]);

  useEffect(() => {
    let isMounted = true;
    fetchSettings()
      .then((payload) => {
        if (!isMounted) {
          return;
        }
        setBackendInfo(payload);
        setSelectedBranch(payload.branches?.[0]?.id || "skeleton");
      })
      .catch((fetchError) => {
        if (!isMounted) {
          return;
        }
        setError(fetchError.message);
      })
      .finally(() => {
        if (isMounted) {
          setLoadingSettings(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => () => {
    revokeQueueRef.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const selectedVideo = useMemo(
    () => videos.find((video) => video.id === selectedVideoId) || null,
    [videos, selectedVideoId]
  );

  function handleUpload(file) {
    const nextItem = createUploadItem(file);
    revokeQueueRef.current.push(nextItem.objectUrl);
    setVideos((current) => {
      const deduped = current.filter((item) => item.id !== nextItem.id);
      return [nextItem, ...deduped];
    });
    setSelectedVideoId(nextItem.id);
    setResult(null);
    setError("");
  }

  function handleRemove(videoId) {
    setVideos((current) => {
      const remaining = current.filter((item) => item.id !== videoId);
      if (selectedVideoId === videoId) {
        setSelectedVideoId(remaining[0]?.id || null);
      }
      return remaining;
    });
    setResult(null);
  }

  async function handlePredict() {
    if (!selectedVideo) {
      return;
    }
    setLoadingPrediction(true);
    setError("");
    setResult(null);
    try {
      const payload = await predictVideo({
        file: selectedVideo.file,
        branch: selectedBranch
      });
      setResult(payload);
    } catch (predictionError) {
      setError(predictionError.message);
    } finally {
      setLoadingPrediction(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="background-glow background-glow-left" />
      <div className="background-glow background-glow-right" />

      <main className="page">
        <Header
          activeSubset={backendInfo?.active_subset}
          poseReady={Boolean(backendInfo?.pose_backend_ready)}
        />

        <ActionBar
          branches={backendInfo?.branches || []}
          selectedBranch={selectedBranch}
          onBranchChange={setSelectedBranch}
          onUpload={handleUpload}
          onPredict={handlePredict}
          canPredict={Boolean(selectedVideo) && !loadingSettings}
          loading={loadingPrediction}
        />

        <section className="content-grid">
          <div className="left-column">
            <VideoPlayer video={selectedVideo} />
            <RecentVideos
              videos={videos}
              selectedId={selectedVideoId}
              onSelect={setSelectedVideoId}
              onRemove={handleRemove}
            />
          </div>

          <div className="right-column">
            <PredictionPanel
              loading={loadingPrediction}
              error={error}
              result={result}
              backendInfo={backendInfo}
            />
          </div>
        </section>
      </main>
    </div>
  );
}
