from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import cv2
import tkinter as tk
from PIL import Image, ImageOps, ImageTk
from tkinter import ttk

from UI.models import ModelOption, PredictionResult, VideoItem, format_duration, format_timestamp


BG = "#050916"
CARD_BG = "#0c1325"
CARD_BG_ALT = "#111a31"
BORDER = "#27314d"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
ACCENT = "#2fb7ff"
ACCENT_ALT = "#bf4cff"
PREDICT_FILL = "#8b5cf6"
PREDICT_OUTLINE = "#d946ef"
SUCCESS = "#38bdf8"
ERROR = "#fb7185"
TRACK = "#24304a"


def configure_ttk_styles(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "Dark.Horizontal.TScale",
        background=BG,
        troughcolor=TRACK,
        bordercolor=TRACK,
        lightcolor=TRACK,
        darkcolor=TRACK,
    )
    style.configure(
        "Dark.TCombobox",
        fieldbackground=CARD_BG_ALT,
        background=CARD_BG_ALT,
        foreground=TEXT,
        arrowcolor=TEXT,
        bordercolor=BORDER,
        relief="flat",
        padding=8,
    )
    style.map(
        "Dark.TCombobox",
        fieldbackground=[("readonly", CARD_BG_ALT)],
        foreground=[("readonly", TEXT)],
        selectbackground=[("readonly", CARD_BG_ALT)],
        selectforeground=[("readonly", TEXT)],
    )
    root.option_add("*TCombobox*Listbox*Background", CARD_BG_ALT)
    root.option_add("*TCombobox*Listbox*Foreground", TEXT)
    root.option_add("*TCombobox*Listbox*selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox*selectForeground", TEXT)


class IconLoader:
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self._cache: dict[tuple[str, tuple[int, int]], ImageTk.PhotoImage] = {}

    def get(self, filename: str, size: tuple[int, int]) -> ImageTk.PhotoImage | None:
        cache_key = (filename, size)
        if cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self.assets_dir / filename
        if not file_path.exists():
            return None

        image = Image.open(file_path).convert("RGBA")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._cache[cache_key] = photo
        return photo


class SectionCard(tk.Frame):
    def __init__(self, master: tk.Misc, *, padding: int = 20, **kwargs):
        super().__init__(
            master,
            bg=CARD_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
            bd=0,
            padx=padding,
            pady=padding,
            **kwargs,
        )


class PageHeader(tk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master, bg=BG)
        tk.Label(
            self,
            text="Sign Language Prediction (Word Level)",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 26),
        ).pack(anchor="w")
        tk.Label(
            self,
            text="Upload a signer video and detect the predicted word with confidence scores.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 12),
        ).pack(anchor="w", pady=(8, 0))


class ActionButton(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        *,
        text: str,
        icon: ImageTk.PhotoImage | None,
        command: Callable[[], None],
        width: int = 220,
        height: int = 54,
        gradient: tuple[str, str] | None = None,
        disabled: bool = False,
        loading: bool = False,
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=BG,
            highlightthickness=0,
            bd=0,
            cursor="hand2" if not disabled else "arrow",
        )
        self.command = command
        self.base_text = text
        self.text = "Predicting..." if loading else text
        self.icon = icon
        self.disabled = disabled
        self.gradient = gradient
        self.width = width
        self.height = height
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", lambda _event: self._redraw(hover=True))
        self.bind("<Leave>", lambda _event: self._redraw(hover=False))
        self._redraw(hover=False)

    def set_state(self, *, disabled: bool, loading: bool = False) -> None:
        self.disabled = disabled
        self.text = "Predicting..." if loading else self.base_text
        self.configure(cursor="arrow" if disabled else "hand2")
        self._redraw(hover=False)

    def _on_click(self, _event) -> None:
        if not self.disabled:
            self.command()

    def _redraw(self, *, hover: bool) -> None:
        self.delete("all")
        if self.disabled:
            fill = "#1b2437"
            outline = "#2a3449"
        elif self.gradient:
            fill = self.gradient[0]
            outline = self.gradient[1]
        else:
            fill = "#121a2c"
            outline = "#37425b"

        if hover and not self.disabled:
            outline = ACCENT if not self.gradient else self.gradient[1]

        self.create_rectangle(
            1,
            1,
            self.width - 1,
            self.height - 1,
            fill=fill,
            outline=outline,
            width=2,
        )
        if self.icon:
            self.create_image(34, self.height / 2, image=self.icon)
        self.create_text(
            self.width / 2 + (16 if self.icon else 0),
            self.height / 2,
            text=self.text,
            fill=MUTED if self.disabled else TEXT,
            font=("Segoe UI Semibold", 12),
        )


class ModelSelector(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        models: list[ModelOption],
        selected_model: tk.StringVar,
        icon: ImageTk.PhotoImage | None,
    ):
        super().__init__(master, bg=BG)
        self.models = models
        self.selected_model = selected_model
        wrapper = tk.Frame(self, bg=CARD_BG_ALT, highlightthickness=1, highlightbackground=BORDER)
        wrapper.pack(fill="x")

        if icon:
            tk.Label(wrapper, image=icon, bg=CARD_BG_ALT).pack(side="left", padx=(14, 10), pady=12)
            self.icon = icon
        else:
            self.icon = None

        self.combo = ttk.Combobox(
            wrapper,
            textvariable=self.selected_model,
            values=[model.label for model in models],
            state="readonly",
            style="Dark.TCombobox",
            width=24,
        )
        self.combo.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=10)

    def get_selected_model(self) -> str:
        current = self.selected_model.get()
        for model in self.models:
            if model.label == current:
                return model.id
        return ""


class ActionBar(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        icons: IconLoader,
        models: list[ModelOption],
        selected_model: tk.StringVar,
        on_upload: Callable[[], None],
        on_predict: Callable[[], None],
    ):
        super().__init__(master, bg=BG)
        self.predict_button = ActionButton(
            self,
            text="Predict",
            icon=icons.get("predict_symbol.png", (20, 20)),
            command=on_predict,
            gradient=(PREDICT_FILL, PREDICT_OUTLINE),
        )
        self.predict_button.pack(side="left")
        self.upload_button = ActionButton(
            self,
            text="Upload",
            icon=icons.get("upload_symbol.png", (18, 18)),
            command=on_upload,
            width=200,
        )
        self.upload_button.pack(side="left", padx=(16, 0))
        self.model_selector = ModelSelector(
            self,
            models=models,
            selected_model=selected_model,
            icon=icons.get("models_symbol.png", (18, 18)),
        )
        self.model_selector.pack(side="right")

    def set_predict_state(self, *, disabled: bool, loading: bool = False) -> None:
        self.predict_button.set_state(disabled=disabled, loading=loading)


class StatusBadge(tk.Label):
    def __init__(self, master: tk.Misc):
        super().__init__(
            master,
            bg=CARD_BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        )

    def set_text(self, message: str, color: str = MUTED) -> None:
        self.configure(text=message, fg=color)


class VideoPlayer(SectionCard):
    PREVIEW_MIN_WIDTH = 900
    PREVIEW_MIN_HEIGHT = 320
    RESIZE_DEBOUNCE_MS = 60

    def __init__(
        self,
        master: tk.Misc,
        *,
        icons: IconLoader,
        on_toggle_fullscreen: Callable[[], None],
    ):
        super().__init__(master, padding=0)
        self.grid_rowconfigure(0, weight=1, minsize=self.PREVIEW_MIN_HEIGHT)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.on_toggle_fullscreen = on_toggle_fullscreen
        self.state = "empty"
        self.video_item: VideoItem | None = None
        self.capture: cv2.VideoCapture | None = None
        self.total_frames = 0
        self.fps = 0.0
        self.duration_seconds = 0.0
        self.current_frame = 0
        self.playing = False
        self._render_handle: str | None = None
        self._current_photo: ImageTk.PhotoImage | None = None
        self._current_frame_image: Image.Image | None = None
        self._loaded_video_id: str | None = None
        self._resize_handle: str | None = None
        self.play_icon = icons.get("play_button.png", (18, 18))
        self.playback_icon = icons.get("playback_button.png", (18, 18))
        self.forward_icon = icons.get("foward_button.png", (18, 18))
        self.fullscreen_icon = icons.get("full_screen_button.png", (18, 18))

        self.footer = tk.Frame(self, bg=CARD_BG)
        self.footer.grid(row=1, column=0, sticky="ew")

        controls = tk.Frame(self.footer, bg="#080d18", padx=18, pady=14)
        controls.pack(fill="x")

        self.play_button = self._small_control(controls, "", self.toggle_play, image=self.play_icon)
        self.play_button.pack(side="left")
        self.rewind_button = self._small_control(controls, "", lambda: self.skip(-2), image=self.playback_icon)
        self.rewind_button.pack(side="left", padx=(14, 0))
        self.forward_button = self._small_control(controls, "", lambda: self.skip(2), image=self.forward_icon)
        self.forward_button.pack(side="left", padx=(14, 0))

        self.time_label = tk.Label(controls, text="00:00 / 00:00", bg="#080d18", fg=TEXT, font=("Segoe UI", 11))
        self.time_label.pack(side="left", padx=(18, 16))

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(
            controls,
            from_=0,
            to=100,
            variable=self.seek_var,
            orient="horizontal",
            style="Dark.Horizontal.TScale",
            command=self._on_seek_drag,
        )
        self.seek_scale.pack(side="left", fill="x", expand=True)

        self.volume_button = self._small_control(controls, "VOL", self._show_audio_notice)
        self.volume_button.pack(side="left", padx=(16, 0))
        self.fullscreen_button = self._small_control(controls, "", self.on_toggle_fullscreen, image=self.fullscreen_icon)
        self.fullscreen_button.pack(side="left", padx=(14, 0))

        self.status_badge = StatusBadge(self.footer)
        self.status_badge.pack(fill="x", padx=20, pady=(0, 14))

        self.preview_container = tk.Frame(
            self,
            bg="#090d17",
            width=self.PREVIEW_MIN_WIDTH,
            height=self.PREVIEW_MIN_HEIGHT,
        )
        self.preview_container.grid(row=0, column=0, sticky="nsew")
        self.preview_container.pack_propagate(False)

        self.preview_canvas = tk.Canvas(
            self.preview_container,
            bg="#090d17",
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", self._schedule_preview_redraw)
        self.preview = self.preview_canvas
        self._preview_text_id = self.preview_canvas.create_text(
            self.PREVIEW_MIN_WIDTH / 2,
            self.PREVIEW_MIN_HEIGHT / 2,
            text="No videos have been uploaded yet",
            fill=MUTED,
            font=("Segoe UI", 14),
            anchor="center",
            justify="center",
        )

        self._set_controls_enabled(False)
        self._set_footer_visibility(False)
        self._render_preview_text("No videos have been uploaded yet")

    def destroy(self) -> None:
        self.stop()
        if self._resize_handle:
            try:
                self.after_cancel(self._resize_handle)
            except Exception:
                pass
        self._release_capture()
        super().destroy()

    def set_video(self, video_item: VideoItem | None) -> None:
        if (
            video_item
            and self.video_item
            and self._loaded_video_id == video_item.id
            and self.capture is not None
        ):
            return

        self.stop()
        self._release_capture()
        self.video_item = video_item
        if not video_item:
            self.state = "empty"
            self._loaded_video_id = None
            self._current_frame_image = None
            self._current_photo = None
            self._set_footer_visibility(False)
            self._render_preview_text("No videos have been uploaded yet")
            self.status_badge.set_text("Select or upload a video to start previewing.")
            self._set_controls_enabled(False)
            self._update_time_label(0.0, 0.0)
            self.seek_var.set(0.0)
            return

        capture = cv2.VideoCapture(str(video_item.path))
        if not capture.isOpened():
            self.state = "error"
            self._current_frame_image = None
            self._current_photo = None
            self._set_footer_visibility(False)
            self._render_preview_text("Unable to open the selected video")
            self.status_badge.set_text("The selected file could not be decoded.", color=ERROR)
            self._set_controls_enabled(False)
            return

        self.capture = capture
        self.total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        self.duration_seconds = video_item.duration_seconds or (self.total_frames / self.fps)
        self.seek_scale.configure(to=max(self.duration_seconds, 0.1))
        self.current_frame = 0
        self._loaded_video_id = video_item.id
        self.state = "ready"
        self._set_footer_visibility(True)
        self._set_controls_enabled(True)
        self.status_badge.set_text(f"Previewing {video_item.name}")
        self._show_frame(0)
        self._schedule_preview_redraw()

    def toggle_play(self) -> None:
        if self.state != "ready" or not self.capture:
            return
        self.playing = not self.playing
        self.play_button.configure(bg="#15213b" if self.playing else "#080d18")
        if self.playing:
            self._playback_loop()

    def stop(self) -> None:
        self.playing = False
        self.play_button.configure(bg="#080d18")
        if self._render_handle:
            try:
                self.after_cancel(self._render_handle)
            except Exception:
                pass
            self._render_handle = None

    def skip(self, seconds: float) -> None:
        if self.state != "ready":
            return
        target = min(max(self.seek_var.get() + seconds, 0.0), self.duration_seconds)
        self._show_frame(target)

    def _show_audio_notice(self) -> None:
        self.status_badge.set_text("This preview focuses on video frames only; audio volume is not available.", color=MUTED)

    def _on_seek_drag(self, value: str) -> None:
        if self.state != "ready" or not self.capture:
            return
        self.playing = False
        self.play_button.configure(bg="#080d18")
        self._show_frame(float(value))

    def _playback_loop(self) -> None:
        if not self.playing or not self.capture:
            return
        next_seconds = min(self.seek_var.get() + (1 / max(self.fps, 1.0)), self.duration_seconds)
        self._show_frame(next_seconds)
        if next_seconds >= self.duration_seconds:
            self.playing = False
            self.play_button.configure(bg="#080d18")
            return
        delay = int(1000 / max(self.fps, 1.0))
        self._render_handle = self.after(delay, self._playback_loop)

    def _show_frame(self, seconds: float) -> None:
        if not self.capture:
            return
        frame_number = int(seconds * self.fps)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            self.status_badge.set_text("Reached the end of the video preview.", color=MUTED)
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_frame_image = Image.fromarray(frame)
        self._render_preview_image()
        self._schedule_preview_redraw()
        self.seek_var.set(min(seconds, self.duration_seconds))
        self.current_frame = frame_number
        self._update_time_label(self.seek_var.get(), self.duration_seconds)

    def _update_time_label(self, current: float, total: float) -> None:
        self.time_label.configure(text=f"{format_duration(current)} / {format_duration(total)}")

    def _render_preview_image(self) -> None:
        if self._current_frame_image is None:
            return

        canvas_width, canvas_height = self._get_preview_size()
        if canvas_width <= 1 or canvas_height <= 1:
            self._schedule_preview_redraw()
            return

        frame_width, frame_height = self._current_frame_image.size
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        render_width = max(1, int(frame_width * scale))
        render_height = max(1, int(frame_height * scale))
        image = self._current_frame_image.resize((render_width, render_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._current_photo = photo

        self.preview_canvas.delete("preview_image")
        self.preview_canvas.itemconfigure(self._preview_text_id, text="")
        self.preview_canvas.create_image(
            canvas_width / 2,
            canvas_height / 2,
            image=photo,
            anchor="center",
            tags="preview_image",
        )

    def _render_preview_text(self, message: str) -> None:
        canvas_width, canvas_height = self._get_preview_size()
        self.preview_canvas.delete("preview_image")
        self.preview_canvas.coords(self._preview_text_id, canvas_width / 2, canvas_height / 2)
        self.preview_canvas.itemconfigure(self._preview_text_id, text=message)

    def _get_preview_size(self) -> tuple[int, int]:
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = self.PREVIEW_MIN_WIDTH
        if canvas_height <= 1:
            canvas_height = self.PREVIEW_MIN_HEIGHT
        return canvas_width, canvas_height

    def _schedule_preview_redraw(self, _event=None) -> None:
        if self._resize_handle:
            try:
                self.after_cancel(self._resize_handle)
            except Exception:
                pass
        self._resize_handle = self.after(self.RESIZE_DEBOUNCE_MS, self._redraw_preview_for_state)

    def _redraw_preview_for_state(self) -> None:
        self._resize_handle = None
        if self.state == "ready" and self._current_frame_image is not None:
            self._set_footer_visibility(True)
            self._render_preview_image()
        elif self.state == "error":
            self._set_footer_visibility(False)
            self._render_preview_text("Unable to open the selected video")
        else:
            self._set_footer_visibility(False)
            self._render_preview_text("No videos have been uploaded yet")

    def refresh_layout_state(self) -> None:
        if self.state == "ready":
            self._set_footer_visibility(True)
            self._schedule_preview_redraw()
        else:
            self._set_footer_visibility(False)

    def _release_capture(self) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (
            self.play_button,
            self.rewind_button,
            self.forward_button,
            self.volume_button,
            self.fullscreen_button,
        ):
            widget.configure(state=state)
        self.seek_scale.state(["!disabled"] if enabled else ["disabled"])

    def _set_footer_visibility(self, visible: bool) -> None:
        is_managed = self.footer.winfo_manager() == "grid"
        if visible and not is_managed:
            self.footer.grid(row=1, column=0, sticky="ew")
        elif not visible and is_managed:
            self.footer.grid_remove()

    @staticmethod
    def _small_control(
        master: tk.Misc,
        label: str,
        command: Callable[[], None],
        *,
        image: ImageTk.PhotoImage | None = None,
    ) -> tk.Button:
        return tk.Button(
            master,
            text=label,
            image=image,
            command=command,
            bg="#080d18",
            fg=TEXT,
            activebackground=CARD_BG_ALT,
            activeforeground=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Segoe UI", 12),
            cursor="hand2",
            compound="center",
            padx=8,
            pady=6,
        )


class VideoCard(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        video: VideoItem,
        selected: bool,
        on_select: Callable[[str], None],
        on_delete: Callable[[str], None],
        delete_icon: ImageTk.PhotoImage | None,
    ):
        super().__init__(
            master,
            bg=CARD_BG_ALT if selected else "#0d1528",
            highlightthickness=1,
            highlightbackground=ACCENT if selected else BORDER,
            width=250,
            height=194,
        )
        self.pack_propagate(False)
        self.video = video
        self._photo: ImageTk.PhotoImage | None = None

        thumb_container = tk.Frame(self, bg="#111827", height=112)
        thumb_container.pack(fill="x")
        thumb_container.pack_propagate(False)
        preview = tk.Label(thumb_container, bg="#111827")
        preview.pack(fill="both", expand=True)

        if video.thumbnail:
            thumb = ImageOps.fit(video.thumbnail.copy(), (248, 112), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(thumb)
            preview.configure(image=self._photo)
        else:
            preview.configure(text="No preview", fg=MUTED, font=("Segoe UI", 11))

        select_button = tk.Button(
            thumb_container,
            text=f"> {format_duration(video.duration_seconds)}",
            command=lambda: on_select(video.id),
            bg="#09111f",
            fg=TEXT,
            activebackground=CARD_BG,
            activeforeground=TEXT,
            bd=0,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        select_button.place(x=12, y=80)

        delete_button = tk.Button(
            thumb_container,
            image=delete_icon,
            command=lambda: on_delete(video.id),
            bg="#eb5166",
            fg=TEXT,
            activebackground="#ff6b81",
            activeforeground=TEXT,
            bd=0,
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=8,
            width=20,
            height=20,
        )
        delete_button.image = delete_icon
        delete_button.place(x=202, y=10, width=36, height=36)

        body = tk.Frame(self, bg=self["bg"], padx=12, pady=10)
        body.pack(fill="both", expand=True)
        title_row = tk.Frame(body, bg=self["bg"])
        title_row.pack(fill="x")
        title_label = tk.Label(title_row, text=video.name, bg=self["bg"], fg=TEXT, font=("Segoe UI Semibold", 12))
        title_label.pack(side="left")
        tk.Button(
            title_row,
            text="...",
            command=lambda: on_select(video.id),
            bg=self["bg"],
            fg=MUTED,
            bd=0,
            relief="flat",
            font=("Segoe UI", 14),
            cursor="hand2",
        ).pack(side="right")
        timestamp_label = tk.Label(
            body,
            text=format_timestamp(video.added_at),
            bg=self["bg"],
            fg=MUTED,
            font=("Segoe UI", 10),
        )
        timestamp_label.pack(anchor="w", pady=(4, 0))
        for widget in (self, thumb_container, preview, body, title_row, title_label, timestamp_label):
            widget.bind("<Button-1>", lambda _event: on_select(video.id))


class RecentVideos(SectionCard):
    SECTION_HEIGHT = 310
    CONTENT_MIN_HEIGHT = 194

    def __init__(
        self,
        master: tk.Misc,
        *,
        icons: IconLoader,
        on_select: Callable[[str], None],
        on_delete: Callable[[str], None],
    ):
        super().__init__(master, padding=20)
        self.configure(height=self.SECTION_HEIGHT)
        self.pack_propagate(False)

        header = tk.Frame(self, bg=CARD_BG)
        header.pack(fill="x")
        icon = icons.get("recent_videos_symbol.png", (18, 18))
        if icon:
            tk.Label(header, image=icon, bg=CARD_BG).pack(side="left")
            self.icon = icon
        tk.Label(header, text="Recent Videos", bg=CARD_BG, fg=TEXT, font=("Segoe UI Semibold", 13)).pack(
            side="left", padx=(10, 0)
        )

        self.scrollbar = tk.Scrollbar(
            self,
            orient="horizontal",
            bg="#172036",
            troughcolor="#0a1020",
            activebackground="#27314d",
            width=16,
            highlightthickness=0,
            bd=0,
            relief="raised",
        )
        self.scrollbar.pack(side="bottom", fill="x", pady=(12, 0))
        self.canvas = tk.Canvas(self, bg=CARD_BG, highlightthickness=0, height=self.CONTENT_MIN_HEIGHT)
        self.canvas.pack(side="top", fill="both", expand=True, pady=(16, 0))
        self.scrollbar.configure(command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.list_frame = tk.Frame(self.canvas, bg=CARD_BG)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.list_frame.bind("<Configure>", self._sync_canvas_window)
        self.canvas.bind("<Configure>", self._sync_canvas_window)

        self._delete_icon = icons.get("delete_symbol.png", (18, 18))
        self.on_select = on_select
        self.on_delete = on_delete

        self.empty_label = tk.Label(
            self.list_frame,
            text="No recent videos yet",
            bg=CARD_BG,
            fg=MUTED,
            font=("Segoe UI", 12),
        )
        self.empty_label.pack(anchor="w", padx=4, pady=72)

    def render(self, videos: list[VideoItem], selected_video_id: str | None) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()

        if not videos:
            self.empty_label = tk.Label(
                self.list_frame,
                text="No recent videos yet",
                bg=CARD_BG,
                fg=MUTED,
                font=("Segoe UI", 12),
            )
            self.empty_label.pack(anchor="w", padx=4, pady=72)
            self.after_idle(self._sync_canvas_window)
            return

        for video in videos:
            card = VideoCard(
                self.list_frame,
                video=video,
                selected=video.id == selected_video_id,
                on_select=self.on_select,
                on_delete=self.on_delete,
                delete_icon=self._delete_icon,
            )
            card.pack(side="left", padx=(0, 18))
        self.after_idle(self._sync_canvas_window)

    def _sync_canvas_window(self, _event=None) -> None:
        canvas_width = max(1, self.canvas.winfo_width())
        content_width = max(canvas_width, self.list_frame.winfo_reqwidth())
        self.canvas.itemconfigure(self.canvas_window, width=content_width, height=self.CONTENT_MIN_HEIGHT)
        self.canvas.configure(scrollregion=(0, 0, content_width, self.CONTENT_MIN_HEIGHT))


class PredictedWordCard(SectionCard):
    def __init__(self, master: tk.Misc):
        super().__init__(master, padding=24)
        self.header = tk.Label(self, text="Results", bg=CARD_BG, fg=TEXT, font=("Segoe UI Semibold", 14))
        self.header.pack(anchor="w")
        self.result_frame = tk.Frame(
            self,
            bg="#111a31",
            highlightbackground=ACCENT_ALT,
            highlightthickness=1,
            padx=24,
            pady=24,
        )
        self.result_frame.pack(fill="both", expand=True, pady=(20, 0))
        self.word_label = tk.Label(
            self.result_frame,
            text="",
            bg="#111a31",
            fg=TEXT,
            font=("Segoe UI Semibold", 38),
        )
        self.word_label.pack(fill="both", expand=True, pady=18)

    def render(self, result: PredictionResult | None) -> None:
        self.word_label.configure(text=result.predicted_word if result else " ")


class ProbabilityBar(tk.Frame):
    def __init__(self, master: tk.Misc, *, label: str, percentage: float):
        super().__init__(master, bg=CARD_BG)
        tk.Label(self, text=label, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 11)).pack(side="left")
        tk.Label(self, text=f"{percentage:.1f}%", bg=CARD_BG, fg=TEXT, font=("Segoe UI", 11)).pack(side="right")
        track = tk.Canvas(self, bg=CARD_BG, height=16, highlightthickness=0)
        track.pack(fill="x", pady=(8, 0))
        track.create_rectangle(0, 4, 320, 12, fill=TRACK, outline=TRACK)
        track.create_rectangle(0, 4, max(2, 3.2 * percentage), 12, fill=ACCENT, outline=ACCENT_ALT)


class PredictionDetailsCard(SectionCard):
    def __init__(self, master: tk.Misc):
        super().__init__(master, padding=20)
        tk.Label(self, text="Prediction Details", bg=CARD_BG, fg=TEXT, font=("Segoe UI Semibold", 14)).pack(anchor="w")
        self.summary = tk.Frame(self, bg=CARD_BG)
        self.summary.pack(fill="x", pady=(16, 12))

        self.word_value = self._summary_block(self.summary, "Predicted Word")
        self.confidence_value = self._summary_block(self.summary, "Confidence", align="e")

        self.separator = tk.Frame(self, bg=BORDER, height=1)
        self.separator.pack(fill="x", pady=(6, 18))
        self.probability_section = tk.Frame(self, bg=CARD_BG)
        self.probability_section.pack(fill="both", expand=True)

    def _summary_block(self, master: tk.Misc, title: str, align: str = "w") -> tk.Label:
        frame = tk.Frame(master, bg=CARD_BG)
        frame.pack(side="left", fill="x", expand=True)
        tk.Label(frame, text=title, bg=CARD_BG, fg=MUTED, font=("Segoe UI", 11)).pack(anchor=align)
        value = tk.Label(frame, text="-", bg=CARD_BG, fg=TEXT, font=("Segoe UI Semibold", 18))
        value.pack(anchor=align, pady=(8, 0))
        return value

    def render(self, status: str, result: PredictionResult | None, error_message: str | None) -> None:
        for child in self.probability_section.winfo_children():
            child.destroy()

        if status == "loading":
            self.word_value.configure(text="Loading...")
            self.confidence_value.configure(text="-")
            tk.Label(
                self.probability_section,
                text="Running prediction...",
                bg=CARD_BG,
                fg=MUTED,
                font=("Segoe UI", 11),
            ).pack(anchor="w")
            return

        if status == "error":
            self.word_value.configure(text="-")
            self.confidence_value.configure(text="-")
            tk.Label(
                self.probability_section,
                text=error_message or "Prediction failed.",
                bg=CARD_BG,
                fg=ERROR,
                justify="left",
                wraplength=360,
                font=("Segoe UI", 11),
            ).pack(anchor="w")
            return

        if not result:
            self.word_value.configure(text="-")
            self.confidence_value.configure(text="-")
            tk.Label(
                self.probability_section,
                text="Prediction results will appear here after you run the model.",
                bg=CARD_BG,
                fg=MUTED,
                justify="left",
                wraplength=360,
                font=("Segoe UI", 11),
            ).pack(anchor="w")
            return

        self.word_value.configure(text=result.predicted_word)
        self.confidence_value.configure(text=f"{result.confidence:.1f}%")
        for item in result.probabilities:
            ProbabilityBar(self.probability_section, label=item.label, percentage=item.probability).pack(
                fill="x", pady=(0, 16)
            )


class PredictionPanel(tk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master, bg=BG)
        self.word_card = PredictedWordCard(self)
        self.word_card.pack(fill="x")
        self.details_card = PredictionDetailsCard(self)
        self.details_card.pack(fill="both", expand=True, pady=(18, 0))

    def render(self, status: str, result: PredictionResult | None, error_message: str | None) -> None:
        self.word_card.render(result if status == "success" else None)
        self.details_card.render(status, result, error_message)


class ToastMessage(tk.Label):
    def __init__(self, master: tk.Misc):
        super().__init__(master, bg=BG, fg=MUTED, font=("Segoe UI", 10), anchor="w")

    def show(self, message: str, *, color: str = MUTED) -> None:
        self.configure(text=message, fg=color)


class BusyOverlay(tk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master, bg=BG)
        self.pack_propagate(False)
        self.label = tk.Label(self, text="Preparing...", bg=BG, fg=TEXT, font=("Segoe UI", 12))
        self.label.pack()


def run_in_thread(worker: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
