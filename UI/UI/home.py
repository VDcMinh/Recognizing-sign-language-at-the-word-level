from __future__ import annotations

import queue
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
import tkinter as tk

import cv2
from PIL import Image

from UI.components import (
    ACCENT,
    BG,
    ERROR,
    MUTED,
    ActionBar,
    IconLoader,
    PageHeader,
    PredictionPanel,
    RecentVideos,
    ToastMessage,
    VideoPlayer,
    run_in_thread,
)
from UI.models import ModelOption, PredictionResult, VideoItem, is_video_file
from UI.services import PredictionService, PredictionServiceUnavailableError


class HomePage(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        assets_dir: Path,
        sample_videos_dir: Path,
        models: list[ModelOption],
        prediction_service: PredictionService,
    ):
        super().__init__(master, bg=BG, padx=32, pady=28)
        self.assets_dir = assets_dir
        self.sample_videos_dir = sample_videos_dir
        self.models = models
        self.prediction_service = prediction_service
        self.icons = IconLoader(assets_dir)

        self.videos: list[VideoItem] = []
        self.selected_video_id: str | None = None
        self.prediction_result: PredictionResult | None = None
        self.prediction_status = "idle"
        self.prediction_error: str | None = None
        self._fullscreen = False
        self._known_paths: set[str] = set()
        self._layout_mode = ""
        self._prediction_queue: queue.Queue[tuple[str, PredictionResult | str]] | None = None

        self.selected_model = tk.StringVar(value=models[0].label if models else "")

        PageHeader(self).pack(fill="x")
        self.action_bar = ActionBar(
            self,
            icons=self.icons,
            models=self.models,
            selected_model=self.selected_model,
            on_upload=self.handle_upload,
            on_predict=self.handle_predict,
        )
        self.action_bar.pack(fill="x", pady=(28, 24))

        self.layout = tk.Frame(self, bg=BG)
        self.layout.pack(fill="both", expand=True)

        self.left_column = tk.Frame(self.layout, bg=BG)
        self.right_column = tk.Frame(self.layout, bg=BG)

        self.player = VideoPlayer(self.left_column, icons=self.icons, on_toggle_fullscreen=self.toggle_fullscreen)
        self.recent_videos = RecentVideos(
            self.left_column,
            icons=self.icons,
            on_select=self.handle_select_video,
            on_delete=self.handle_delete_video,
        )
        self.prediction_panel = PredictionPanel(self.right_column)

        self.toast = ToastMessage(self)
        self.toast.pack(fill="x", pady=(14, 0))

        self._configure_responsive_layout()
        self.bind("<Configure>", self._handle_resize)
        self.after(0, lambda: self._apply_layout_mode("mobile" if self.winfo_width() < 768 else "desktop"))
        self._render()

    def _configure_responsive_layout(self) -> None:
        self.layout.grid_columnconfigure(0, weight=3)
        self.layout.grid_columnconfigure(1, weight=2)
        self.layout.grid_rowconfigure(0, weight=1)

    def _handle_resize(self, event) -> None:
        mode = "mobile" if event.width < 768 else "desktop"
        self._apply_layout_mode(mode)

    def _apply_layout_mode(self, mode: str) -> None:
        if mode == self._layout_mode:
            return

        self._layout_mode = mode

        for widget in (self.left_column, self.right_column, self.player, self.recent_videos, self.prediction_panel):
            widget.grid_forget()
            widget.pack_forget()

        if mode == "mobile":
            self.layout.grid_columnconfigure(0, weight=1)
            self.layout.grid_columnconfigure(1, weight=0)
            self.left_column.grid(row=0, column=0, sticky="nsew")
            self.right_column.grid(row=1, column=0, sticky="ew", pady=(18, 0))

            self.player.pack(fill="x")
            self.recent_videos.pack(fill="x", pady=(18, 0))
            self.prediction_panel.pack(fill="x")
        else:
            self.layout.grid_columnconfigure(0, weight=3)
            self.layout.grid_columnconfigure(1, weight=2)
            self.left_column.grid(row=0, column=0, sticky="nsew")
            self.right_column.grid(row=0, column=1, sticky="nsew", padx=(24, 0))

            self.recent_videos.pack(side="bottom", fill="x")
            self.player.pack(side="top", fill="both", expand=True, pady=(0, 18))
            self.prediction_panel.pack(fill="both", expand=True)

    def handle_upload(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select a sign language video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        if not is_video_file(file_path):
            self.toast.show("The selected file is not a supported video format.", color=ERROR)
            return

        resolved = str(Path(file_path).resolve())
        if resolved in self._known_paths:
            self.toast.show("That video is already in Recent Videos.", color=MUTED)
            return

        video = self._create_video_item(Path(file_path))
        if not video:
            self.toast.show("The selected video could not be decoded.", color=ERROR)
            return

        self.videos.insert(0, video)
        self._known_paths.add(resolved)
        self.selected_video_id = video.id
        self._reset_prediction()
        self._render()
        self.toast.show(f"Added {video.name} to Recent Videos.", color=ACCENT)

    def handle_select_video(self, video_id: str) -> None:
        if self.selected_video_id == video_id:
            return
        self.selected_video_id = video_id
        self._reset_prediction()
        self._render()
        current = self._selected_video
        if current:
            self.toast.show(f"Selected {current.name}.", color=ACCENT)

    def handle_delete_video(self, video_id: str) -> None:
        if not self.videos:
            return

        deleting_selected = self.selected_video_id == video_id
        removed: VideoItem | None = None
        remaining: list[VideoItem] = []
        for video in self.videos:
            if video.id == video_id:
                removed = video
            else:
                remaining.append(video)

        if removed is None:
            return

        self.videos = remaining
        self._known_paths.discard(str(removed.path.resolve()))

        if deleting_selected:
            self.selected_video_id = self.videos[0].id if self.videos else None
            self._reset_prediction()

        self._render()
        self.toast.show(f"Removed {removed.name}.", color=MUTED)

    def handle_predict(self) -> None:
        video = self._selected_video
        model_id = self.action_bar.model_selector.get_selected_model()
        if not video or not model_id or self.prediction_status == "loading":
            return

        self.prediction_status = "loading"
        self.prediction_error = None
        self.prediction_result = None
        self._render()
        self.toast.show(f"Running prediction for {video.name} with {self.selected_model.get()}...", color=ACCENT)

        def worker() -> None:
            try:
                result = self.prediction_service.predict(video.path, model_id)
            except (PredictionServiceUnavailableError, ValueError, RuntimeError) as exc:
                if self._prediction_queue is not None:
                    self._prediction_queue.put(("error", str(exc)))
                return
            if self._prediction_queue is not None:
                self._prediction_queue.put(("success", result))

        self._prediction_queue = queue.Queue()
        run_in_thread(worker)
        self.after(50, self._poll_prediction_queue)

    def _finish_prediction_success(self, result: PredictionResult) -> None:
        self.prediction_status = "success"
        self.prediction_result = result
        self.prediction_error = None
        self._render()
        self.toast.show("Prediction completed successfully.", color=ACCENT)

    def _finish_prediction_error(self, message: str) -> None:
        self.prediction_status = "error"
        self.prediction_result = None
        self.prediction_error = message
        self._render()
        self.toast.show(message, color=ERROR)

    def _poll_prediction_queue(self) -> None:
        if self._prediction_queue is None:
            return
        try:
            status, payload = self._prediction_queue.get_nowait()
        except queue.Empty:
            if self.prediction_status == "loading":
                self.after(50, self._poll_prediction_queue)
            return

        self._prediction_queue = None
        if status == "success" and isinstance(payload, PredictionResult):
            self._finish_prediction_success(payload)
        else:
            self._finish_prediction_error(str(payload))

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.winfo_toplevel().attributes("-fullscreen", self._fullscreen)
        self.after(120, self.player.refresh_layout_state)

    def _create_video_item(self, path: Path) -> VideoItem | None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None

        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1
        capture.release()
        if not ok:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thumbnail = Image.fromarray(frame)
        duration_seconds = max(0.1, float(frame_count / fps))
        return VideoItem(
            id=str(uuid.uuid4()),
            path=path,
            name=path.stem,
            duration_seconds=duration_seconds,
            added_at=datetime.now(),
            thumbnail=thumbnail,
        )

    def _reset_prediction(self) -> None:
        self.prediction_result = None
        self.prediction_status = "idle"
        self.prediction_error = None

    @property
    def _selected_video(self) -> VideoItem | None:
        for video in self.videos:
            if video.id == self.selected_video_id:
                return video
        return None

    def _render(self) -> None:
        self.player.set_video(self._selected_video)
        self.recent_videos.render(self.videos, self.selected_video_id)
        self.prediction_panel.render(self.prediction_status, self.prediction_result, self.prediction_error)
        predict_disabled = not self._selected_video or not self.action_bar.model_selector.get_selected_model()
        self.action_bar.set_predict_state(
            disabled=predict_disabled or self.prediction_status == "loading",
            loading=self.prediction_status == "loading",
        )
