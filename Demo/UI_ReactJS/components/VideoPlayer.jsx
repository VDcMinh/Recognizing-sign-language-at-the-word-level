import { useEffect, useRef, useState } from "react";
import { formatDuration } from "../utils/formatters";

export default function VideoPlayer({ video, onToast }) {
  const videoRef = useRef(null);
  const shellRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const element = videoRef.current;
    if (!element) {
      return undefined;
    }

    const syncState = () => {
      setCurrentTime(element.currentTime || 0);
      setDuration(element.duration || 0);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(element.duration || 0);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    element.addEventListener("loadedmetadata", syncState);
    element.addEventListener("timeupdate", syncState);
    element.addEventListener("durationchange", syncState);
    element.addEventListener("ended", handleEnded);
    element.addEventListener("play", handlePlay);
    element.addEventListener("pause", handlePause);

    return () => {
      element.removeEventListener("loadedmetadata", syncState);
      element.removeEventListener("timeupdate", syncState);
      element.removeEventListener("durationchange", syncState);
      element.removeEventListener("ended", handleEnded);
      element.removeEventListener("play", handlePlay);
      element.removeEventListener("pause", handlePause);
    };
  }, [video?.id]);

  useEffect(() => {
    const element = videoRef.current;
    if (!element) {
      return;
    }

    element.pause();
    element.currentTime = 0;
    element.muted = true;
    setIsMuted(true);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(video?.durationSeconds || 0);
  }, [video?.id, video?.durationSeconds]);

  const togglePlay = async () => {
    const element = videoRef.current;
    if (!element || !video) {
      return;
    }

    if (element.paused) {
      await element.play();
      return;
    }

    element.pause();
  };

  const skipBy = (seconds) => {
    const element = videoRef.current;
    if (!element || !video) {
      return;
    }

    const nextTime = Math.min(
      Math.max((element.currentTime || 0) + seconds, 0),
      element.duration || duration || 0
    );
    element.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  const handleSeek = (event) => {
    const element = videoRef.current;
    const nextTime = Number(event.target.value);

    if (!element || !video) {
      return;
    }

    element.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  const toggleMute = () => {
    const element = videoRef.current;
    if (!element) {
      return;
    }

    const nextMuted = !element.muted;
    element.muted = nextMuted;
    setIsMuted(nextMuted);
    onToast(nextMuted ? "Preview audio muted." : "Preview audio enabled.");
  };

  const toggleFullscreen = async () => {
    const shell = shellRef.current;
    if (!shell) {
      return;
    }

    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }

    await shell.requestFullscreen();
  };

  return (
    <section className="video-player card" ref={shellRef}>
      <div className="video-stage">
        {video ? (
          <video
            ref={videoRef}
            src={video.src}
            className="video-element"
            playsInline
            muted={isMuted}
            preload="metadata"
          />
        ) : (
          <div className="video-empty-state">
            <h2>No videos have been uploaded yet</h2>
            <p>Select a sample or upload your own sign language clip to begin.</p>
          </div>
        )}
      </div>

      <div className="video-controls">
        <div className="video-controls-row">
          <button type="button" className="mini-button" onClick={togglePlay} disabled={!video}>
            {isPlaying ? (
              <span className="pause-glyph" aria-hidden="true">
                II
              </span>
            ) : (
              <img src="/elements/play_button.png" alt="" aria-hidden="true" />
            )}
          </button>
          <button type="button" className="mini-button" onClick={() => skipBy(-2)} disabled={!video}>
            <img src="/elements/playback_button.png" alt="" aria-hidden="true" />
          </button>
          <button type="button" className="mini-button" onClick={() => skipBy(2)} disabled={!video}>
            <img src="/elements/foward_button.png" alt="" aria-hidden="true" />
          </button>

          <div className="timeline-readout">
            {formatDuration(currentTime)} / {formatDuration(duration)}
          </div>

          <input
            className="timeline-slider"
            type="range"
            min="0"
            max={Math.max(duration, 0.1)}
            step="0.01"
            value={Math.min(currentTime, duration)}
            onChange={handleSeek}
            disabled={!video}
          />

          <button type="button" className="mini-button text-button" onClick={toggleMute} disabled={!video}>
            {isMuted ? "VOL" : "ON"}
          </button>
          <button type="button" className="mini-button" onClick={toggleFullscreen} disabled={!video}>
            <img src="/elements/full_screen_button.png" alt="" aria-hidden="true" />
          </button>
        </div>

        <div className="player-status">
          {video
            ? `Previewing ${video.name}`
            : "Select a recent video card or upload a new video to start previewing."}
        </div>
      </div>
    </section>
  );
}
