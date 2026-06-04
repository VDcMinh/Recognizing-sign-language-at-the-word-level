from __future__ import annotations

import html
import streamlit as st

from ui_core.pipeline import run_prediction_pipeline
from ui_core.state import (
    add_recent_video,
    get_recent_videos,
    get_selected_video_record,
    init_session_state,
    remove_recent_video,
    select_recent_video,
    set_last_error,
    set_last_prediction,
    toggle_recent_view,
)
from ui_core.styles import inject_global_styles
from ui_core.video_utils import get_video_metadata, save_uploaded_video, validate_video_file
from ui_core.inference import is_mock_mode_enabled
from ui_core.model_registry import get_model_option_names


st.set_page_config(
    page_title="Word-level Sign Language Recognition Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _render_notice(message: str, *, kind: str = "info") -> None:
    escaped = html.escape(message)
    st.markdown(
        f"<div class='notice notice-{kind}'>{escaped}</div>",
        unsafe_allow_html=True,
    )


def _render_recent_video_card(record: dict, *, selected: bool) -> None:
    metadata = record["metadata"]
    file_label = html.escape(record["filename"])
    meta_line = " • ".join(
        [
            str(metadata.get("duration_label") or "--:--"),
            str(metadata.get("extension") or "--"),
            str(metadata.get("file_size_label") or "--"),
        ]
    )
    st.markdown(
        f"""
        <div class="recent-card" style="border-color:{'#D98CA6' if selected else '#E8DEE3'};">
            <div class="recent-thumb">
                <div class="recent-delete">&#10005;</div>
                <div class="recent-play">&#9654;</div>
            </div>
            <p class="recent-name">{file_label}</p>
            <p class="recent-meta">{html.escape(meta_line)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_card(
    prediction: dict | None,
    last_error: str | None,
    *,
    has_selected_video: bool,
) -> None:
    st.markdown(
        """
        <div class="ui-card">
            <div class="ui-section-header">
                <h3 class="ui-section-title"><span>&#9634;</span>Result</h3>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if prediction and prediction.get("status") == "success":
        word = html.escape(f"\"{prediction['predicted_word']}\"")
        st.markdown(
            f"""
            <div class="result-preview">
                <p class="result-word">{word}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if last_error:
            empty_message = last_error
        elif has_selected_video:
            empty_message = "No prediction yet"
        else:
            empty_message = "No video uploaded yet"
        st.markdown(
            f"""
            <div class="result-preview">
                <p class="result-empty">{html.escape(empty_message)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_prediction_details_card(prediction: dict | None) -> None:
    st.markdown(
        """
        <div class="ui-card">
            <div class="ui-section-header">
                <h3 class="ui-section-title"><span>&#9673;</span>Prediction Details</h3>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if prediction and prediction.get("status") == "success":
        confidence_pct = prediction["confidence"] * 100
        st.markdown(
            f"""
            <div class="details-grid">
                <div class="details-block">
                    <p class="details-label">Predicted Word</p>
                    <p class="details-value">{html.escape(prediction['predicted_word'])}</p>
                </div>
                <div class="details-block">
                    <p class="details-label">Confidence</p>
                    <p class="details-value details-accent">{confidence_pct:.2f}%</p>
                </div>
            </div>
            <div class="dashed-divider"></div>
            <p class="details-label" style="margin-bottom:0.9rem;">Probabilities</p>
            """,
            unsafe_allow_html=True,
        )
        probability_rows = []
        for item in prediction.get("topk", []):
            probability_rows.append(
                f"""
                <div class="prob-row">
                    <div class="prob-word">{html.escape(str(item['word']))}</div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width:{float(item['prob']) * 100:.2f}%;"></div>
                    </div>
                    <div class="prob-value">{float(item['prob']) * 100:.0f}%</div>
                </div>
                """
            )
        st.markdown("".join(probability_rows), unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <p class="recent-meta" style="font-size:0.98rem; line-height:1.75;">
                Upload a video and click Predict to see results.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _handle_upload(uploaded_file) -> None:
    if uploaded_file is None:
        return

    is_valid, validation_message = validate_video_file(uploaded_file)
    if not is_valid:
        set_last_error(validation_message)
        _render_notice(validation_message, kind="error")
        return

    file_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("last_upload_signature") == file_signature:
        return

    saved_path = save_uploaded_video(uploaded_file)
    metadata = get_video_metadata(saved_path)
    add_recent_video(saved_path, metadata)
    set_last_prediction(None)
    set_last_error(None)
    st.session_state["last_upload_signature"] = file_signature
    _render_notice(
        f"Video uploaded successfully: {metadata['filename']} ({metadata['file_size_label']}).",
        kind="info",
    )


def _render_recent_videos_section() -> None:
    st.markdown(
        """
        <div class="ui-card">
            <div class="ui-section-header">
                <h3 class="ui-section-title"><span>&#9685;</span>Recent Videos</h3>
                <div class="ui-muted-action">View All &#8594;</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "View All",
        key="toggle_recent_videos",
        use_container_width=False,
        on_click=toggle_recent_view,
    )

    recent_videos = get_recent_videos()
    if not recent_videos:
        st.markdown(
            """
            <p class="recent-meta" style="font-size:0.98rem; line-height:1.75;">
                No videos yet. Upload a clip to start building your recent list.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    show_all = bool(st.session_state.get("show_all_recent", False))
    display_items = recent_videos if show_all else recent_videos[:3]
    selected_record = get_selected_video_record()
    selected_video_id = selected_record["id"] if selected_record else None

    for start_index in range(0, len(display_items), 3):
        row_items = display_items[start_index : start_index + 3]
        columns = st.columns(len(row_items))
        for column, record in zip(columns, row_items):
            with column:
                _render_recent_video_card(record, selected=record["id"] == selected_video_id)
                preview_col, delete_col = st.columns(2)
                with preview_col:
                    if st.button("Preview", key=f"preview_{record['id']}", use_container_width=True):
                        select_recent_video(record["id"])
                        st.rerun()
                with delete_col:
                    if st.button("Delete", key=f"delete_{record['id']}", use_container_width=True):
                        remove_recent_video(record["id"])
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _run_prediction_if_requested(predict_clicked: bool) -> None:
    if not predict_clicked:
        return

    selected_video = get_selected_video_record()
    if selected_video is None:
        set_last_prediction(None)
        set_last_error("No video uploaded yet")
        return

    with st.status("Analyzing sign language video...", expanded=True) as status:
        def push_step(message: str) -> None:
            status.write(message)

        result = run_prediction_pipeline(
            video_path=selected_video["path"],
            model_name=st.session_state["selected_model"],
            status_callback=push_step,
        )

        if result.get("status") == "success":
            status.update(label="Prediction complete", state="complete")
            set_last_prediction(result)
            set_last_error(None)
        else:
            status.update(label="Prediction failed", state="error")
            set_last_prediction(None)
            set_last_error(str(result.get("message", "Prediction failed.")))


inject_global_styles()
init_session_state()

st.markdown(
    """
    <div class="ui-shell-header">
        <div>
            <div class="ui-kicker">&#10024; Modern Streamlit Demo</div>
            <div class="ui-hero">
                <h1 class="ui-title">Word-level Sign Language Recognition Demo</h1>
                <p class="ui-subtitle">
                    Upload a sign language video, switch between Skeleton and Skeleton + Fusion,
                    and inspect the predicted word with top-5 probabilities in a soft premium dashboard.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

header_left, header_spacer, header_right = st.columns([0.22, 0.5, 0.28], gap="small")
with header_left:
    predict_clicked = st.button("Predict", key="predict_button", use_container_width=True)
with header_right:
    st.selectbox(
        "Models",
        options=get_model_option_names(),
        key="selected_model",
        label_visibility="collapsed",
    )

_run_prediction_if_requested(predict_clicked)

left_col, right_col = st.columns([0.68, 0.32], gap="large")

with left_col:
    st.markdown(
        """
        <div class="upload-intro">
            <div class="meta-chip">&#10022; Upload and preview your sign clip</div>
            <div class="upload-icon">&#9729;</div>
            <h2 class="upload-title">Upload your video</h2>
            <p class="upload-copy">
                Drag and drop a video file here, or browse to choose a clip for word-level sign language recognition.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload your video",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )
    _handle_upload(uploaded_file)
    st.markdown(
        """
        <div class="support-copy">
            Supports MP4, MOV, AVI • Max size 500MB • Max duration 2 min
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_video = get_selected_video_record()
    if selected_video is not None:
        metadata = selected_video["metadata"]
        st.markdown(
            f"""
            <div class="meta-chip">
                &#11049; {html.escape(str(metadata.get('duration_label', '--:--')))}
                • {html.escape(str(metadata.get('extension', '--')))}
                • {html.escape(str(metadata.get('file_size_label', '--')))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.video(selected_video["path"])

    _render_recent_videos_section()

with right_col:
    last_prediction = st.session_state.get("last_prediction")
    last_error = st.session_state.get("last_error")
    selected_video = get_selected_video_record()

    if is_mock_mode_enabled():
        _render_notice(
            "Demo/mock mode is ON. The app stays usable without real checkpoints, and every saved prediction log records demo_mode=true.",
            kind="info",
        )

    if last_prediction and last_prediction.get("warnings"):
        for warning in last_prediction["warnings"]:
            _render_notice(str(warning), kind="info")

    if last_error:
        _render_notice(last_error, kind="error")

    _render_result_card(
        last_prediction,
        last_error,
        has_selected_video=selected_video is not None,
    )
    _render_prediction_details_card(last_prediction)
