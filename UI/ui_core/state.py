from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


DEFAULT_MODEL_NAME = "Skeleton"
MAX_RECENT_VIDEOS = 12


def init_session_state() -> None:
    st.session_state.setdefault("recent_videos", [])
    st.session_state.setdefault("selected_video", None)
    st.session_state.setdefault("selected_model", DEFAULT_MODEL_NAME)
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("last_upload_signature", None)
    st.session_state.setdefault("show_all_recent", False)


def get_recent_videos() -> list[dict[str, Any]]:
    return list(st.session_state.get("recent_videos", []))


def get_selected_video_record() -> dict[str, Any] | None:
    selected_video = st.session_state.get("selected_video")
    if selected_video is None:
        return None
    for record in get_recent_videos():
        if record["id"] == selected_video:
            return record
    return None


def add_recent_video(video_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    path = Path(video_path)
    record = {
        "id": path.stem,
        "path": path.as_posix(),
        "filename": str(metadata.get("filename", path.name)),
        "metadata": metadata,
    }

    existing = [
        item for item in get_recent_videos() if item["path"] != record["path"]
    ]
    updated = [record, *existing][:MAX_RECENT_VIDEOS]
    st.session_state["recent_videos"] = updated
    st.session_state["selected_video"] = record["id"]
    return record


def select_recent_video(video_id: str) -> None:
    st.session_state["selected_video"] = video_id


def remove_recent_video(video_id: str) -> None:
    updated = [
        item for item in get_recent_videos() if item["id"] != video_id
    ]
    st.session_state["recent_videos"] = updated
    if st.session_state.get("selected_video") == video_id:
        st.session_state["selected_video"] = updated[0]["id"] if updated else None
        st.session_state["last_prediction"] = None
        st.session_state["last_error"] = None


def set_last_prediction(prediction: dict[str, Any] | None) -> None:
    st.session_state["last_prediction"] = prediction
    st.session_state["last_error"] = None if prediction else st.session_state.get("last_error")


def set_last_error(message: str | None) -> None:
    st.session_state["last_error"] = message


def toggle_recent_view() -> None:
    st.session_state["show_all_recent"] = not bool(
        st.session_state.get("show_all_recent", False)
    )
