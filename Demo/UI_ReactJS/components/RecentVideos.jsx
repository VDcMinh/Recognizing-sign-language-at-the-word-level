import { formatDuration, formatTimestamp } from "../utils/formatters";

export default function RecentVideos({
  videos,
  selectedVideoId,
  onSelect,
  onDelete,
  loading
}) {
  return (
    <section className="recent-videos card">
      <div className="section-title">
        <img src="/elements/recent_videos_symbol.png" alt="" aria-hidden="true" />
        <h2>Recent Videos</h2>
      </div>

      <div className="recent-videos-track">
        {videos.length === 0 ? (
          <div className="empty-recent">
            {loading ? "Loading sample videos..." : "No recent videos yet"}
          </div>
        ) : (
          videos.map((video) => (
            <article
              key={video.id}
              className={`video-card ${video.id === selectedVideoId ? "is-selected" : ""}`}
              onClick={() => onSelect(video.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(video.id);
                }
              }}
              role="button"
              tabIndex={0}
            >
              <div className="video-card-thumb">
                {video.thumbnail ? (
                  <img src={video.thumbnail} alt={`${video.name} thumbnail`} />
                ) : (
                  <div className="video-card-thumb-fallback">No preview</div>
                )}

                <button
                  type="button"
                  className="duration-chip"
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelect(video.id);
                  }}
                >
                  {formatDuration(video.durationSeconds)}
                </button>

                <button
                  type="button"
                  className="delete-chip"
                  onClick={(event) => {
                    event.stopPropagation();
                    onDelete(video.id);
                  }}
                  aria-label={`Delete ${video.name}`}
                >
                  <img src="/elements/delete_symbol.png" alt="" aria-hidden="true" />
                </button>
              </div>

              <div className="video-card-body">
                <div className="video-card-heading">
                  <h3>{video.name}</h3>
                  <button type="button" className="more-chip" onClick={() => onSelect(video.id)}>
                    ...
                  </button>
                </div>
                <p>{formatTimestamp(video.addedAt)}</p>
                <span className="source-chip">{video.kind === "sample" ? "Sample Video" : "Uploaded Video"}</span>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}
