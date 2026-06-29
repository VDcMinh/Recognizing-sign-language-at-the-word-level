import { useEffect, useMemo, useRef, useState } from "react";
import ActionBar from "./components/ActionBar";
import Header from "./components/Header";
import PredictionPanel from "./components/PredictionPanel";
import RecentVideos from "./components/RecentVideos";
import Toast from "./components/Toast";
import VideoPlayer from "./components/VideoPlayer";
import { models } from "./data/models";
import { predictWordLevel } from "./services/predictionService";
import { createVideoItem } from "./utils/videoItems";

const SUPPORTED_EXTENSIONS = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]);

function isSupportedVideoFile(fileName) {
  const extension = fileName.slice(fileName.lastIndexOf(".")).toLowerCase();
  return SUPPORTED_EXTENSIONS.has(extension);
}

export default function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState(models[0]?.id || "");
  const [predictionStatus, setPredictionStatus] = useState("idle");
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState(null);
  const [toast, setToast] = useState(null);
  const toastTimerRef = useRef(null);
  const videosRef = useRef([]);

  const selectedVideo = useMemo(
    () => videos.find((item) => item.id === selectedVideoId) || null,
    [videos, selectedVideoId]
  );

  useEffect(() => {
    videosRef.current = videos;
  }, [videos]);

  const showToast = (message, tone = "info") => {
    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current);
    }

    setToast({
      id: Date.now(),
      message,
      tone
    });

    toastTimerRef.current = window.setTimeout(() => {
      setToast(null);
    }, 3200);
  };

  useEffect(() => () => {
    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current);
    }
    videosRef.current.forEach((video) => {
      if (video.shouldRevoke) {
        URL.revokeObjectURL(video.src);
      }
    });
  }, []);

  const resetPrediction = () => {
    setPredictionStatus("idle");
    setPredictionResult(null);
    setPredictionError(null);
  };

  const handleSelectVideo = (videoId) => {
    if (videoId === selectedVideoId) {
      return;
    }

    setSelectedVideoId(videoId);
    resetPrediction();

    const video = videos.find((item) => item.id === videoId);
    if (video) {
      showToast(`Selected ${video.name}.`);
    }
  };

  const handleDeleteVideo = (videoId) => {
    const target = videos.find((item) => item.id === videoId);
    if (!target) {
      return;
    }

    if (target.shouldRevoke) {
      URL.revokeObjectURL(target.src);
    }

    const remaining = videos.filter((item) => item.id !== videoId);
    setVideos(remaining);

    if (videoId === selectedVideoId) {
      setSelectedVideoId(remaining[0]?.id || null);
      resetPrediction();
    }

    showToast(`Removed ${target.name}.`);
  };

  const handleUpload = async (files) => {
    const supportedFiles = files.filter((file) => isSupportedVideoFile(file.name));

    if (supportedFiles.length === 0) {
      showToast("The selected files are not supported video formats.", "error");
      return;
    }

    const existingKeys = new Set(videos.map((item) => item.sourceKey));

    try {
      const newItems = [];

      for (const file of supportedFiles) {
        const sourceKey = `upload:${file.name}:${file.size}:${file.lastModified}`;
        if (existingKeys.has(sourceKey)) {
          continue;
        }

        const src = URL.createObjectURL(file);
        const item = await createVideoItem({
          src,
          name: file.name,
          sourceKey,
          kind: "upload",
          shouldRevoke: true
        });
        newItems.push(item);
        existingKeys.add(sourceKey);
      }

      if (newItems.length === 0) {
        showToast("Those videos are already in Recent Videos.");
        return;
      }

      const orderedItems = [...newItems].reverse();

      setVideos((current) => [...orderedItems, ...current]);
      setSelectedVideoId(orderedItems[0].id);
      resetPrediction();
      showToast(`Added ${newItems.length} video(s) to Recent Videos.`);
    } catch (error) {
      showToast(error.message || "The selected video could not be decoded.", "error");
    }
  };

  const handlePredict = async () => {
    if (!selectedVideo || !selectedModelId || predictionStatus === "loading") {
      return;
    }

    const model = models.find((item) => item.id === selectedModelId);
    setPredictionStatus("loading");
    setPredictionError(null);
    setPredictionResult(null);
    showToast(`Running prediction for ${selectedVideo.name} with ${model?.label || selectedModelId}...`);

    try {
      const result = await predictWordLevel(selectedVideo, selectedModelId);
      setPredictionStatus("success");
      setPredictionResult(result);
      showToast("Prediction completed successfully.");
    } catch (error) {
      setPredictionStatus("error");
      setPredictionError(error.message || "Prediction failed.");
      showToast(error.message || "Prediction failed.", "error");
    }
  };

  return (
    <div className="app-shell">
      <div className="background-glow background-glow-left" />
      <div className="background-glow background-glow-right" />

      <main className="page">
        <Header />

        <ActionBar
          models={models}
          selectedModelId={selectedModelId}
          onModelChange={setSelectedModelId}
          onPredict={handlePredict}
          onUpload={handleUpload}
          predictDisabled={!selectedVideo || !selectedModelId || predictionStatus === "loading"}
          loading={predictionStatus === "loading"}
        />

        <section className="content-grid">
          <div className="left-column">
            <VideoPlayer video={selectedVideo} onToast={showToast} />
            <RecentVideos
              videos={videos}
              selectedVideoId={selectedVideoId}
              onSelect={handleSelectVideo}
              onDelete={handleDeleteVideo}
              loading={false}
            />
          </div>

          <div className="right-column">
            <PredictionPanel
              status={predictionStatus}
              result={predictionResult}
              error={predictionError}
            />
          </div>
        </section>

        <Toast toast={toast} />
      </main>
    </div>
  );
}
