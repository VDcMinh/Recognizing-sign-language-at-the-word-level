from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-cream: #FAF7F5;
            --bg-shell: #FFFFFF;
            --text-main: #1F1D2B;
            --text-soft: #77717C;
            --border-soft: #E8DEE3;
            --panel-dark: #241D2B;
            --accent-plum: #9B3F68;
            --accent-pink: #D98CA6;
            --accent-peach: #F2B8C6;
            --shadow-soft: 0 22px 60px rgba(36, 29, 43, 0.10);
            --shadow-card: 0 14px 35px rgba(36, 29, 43, 0.08);
            --radius-lg: 28px;
            --radius-md: 20px;
            --radius-pill: 999px;
        }

        html, body, [class*="css"] {
            font-family: Inter, Poppins, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text-main);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(242, 184, 198, 0.55), transparent 30%),
                radial-gradient(circle at top right, rgba(217, 140, 166, 0.32), transparent 22%),
                linear-gradient(180deg, #F7F2EF 0%, #FAF7F5 100%);
        }

        .main .block-container {
            max-width: 1380px;
            padding-top: 1.75rem;
            padding-bottom: 2rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(232, 222, 227, 0.9);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-soft);
        }

        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: var(--text-main);
        }

        div.stButton > button {
            border-radius: var(--radius-pill);
            border: none;
            background: linear-gradient(135deg, var(--accent-plum) 0%, var(--accent-pink) 52%, var(--accent-peach) 100%);
            color: #FFFFFF;
            font-weight: 700;
            padding: 0.8rem 1.25rem;
            box-shadow: 0 12px 28px rgba(155, 63, 104, 0.25);
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, #8C345C 0%, #C77294 52%, #EDAFBF 100%);
        }

        div[data-baseweb="select"] > div {
            min-height: 54px;
            border-radius: var(--radius-pill);
            border: 1px solid var(--border-soft);
            background: #FFFFFF;
            box-shadow: none;
        }

        div[data-baseweb="select"] span {
            color: var(--text-main);
            font-weight: 600;
        }

        [data-testid="stFileUploader"] {
            background: linear-gradient(180deg, #241D2B 0%, #1F1A27 100%);
            border-radius: var(--radius-lg);
            padding: 0.4rem;
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stFileUploaderDropzone"] {
            background:
                radial-gradient(circle at bottom left, rgba(217, 140, 166, 0.18), transparent 28%),
                linear-gradient(180deg, rgba(36, 29, 43, 0.98) 0%, rgba(31, 26, 39, 0.98) 100%);
            border: 1.5px dashed rgba(242, 184, 198, 0.9);
            border-radius: var(--radius-lg);
            min-height: 320px;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #F8EDF1;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] > div {
            color: #E7D5DE;
        }

        [data-testid="stFileUploaderDropzone"] button {
            border-radius: var(--radius-pill);
            background: linear-gradient(135deg, var(--accent-plum) 0%, var(--accent-pink) 100%);
            color: #FFFFFF;
            border: none;
            box-shadow: 0 12px 28px rgba(155, 63, 104, 0.22);
        }

        .ui-shell-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.2rem 0 1rem;
        }

        .ui-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.85rem;
            background: rgba(242, 184, 198, 0.22);
            border: 1px solid rgba(217, 140, 166, 0.35);
            color: var(--accent-plum);
            border-radius: var(--radius-pill);
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        .ui-hero {
            padding: 0.2rem 0 0.8rem;
        }

        .ui-title {
            font-size: clamp(2rem, 2.7vw, 3rem);
            line-height: 1.05;
            font-weight: 800;
            margin: 0.4rem 0 0.75rem;
            color: var(--text-main);
        }

        .ui-subtitle {
            max-width: 720px;
            margin: 0;
            color: var(--text-soft);
            font-size: 1rem;
            line-height: 1.7;
        }

        .ui-card {
            background: #FFFFFF;
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-soft);
            box-shadow: var(--shadow-card);
            padding: 1.25rem;
        }

        .ui-section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 1rem;
        }

        .ui-section-title {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0;
            font-size: 1.1rem;
            font-weight: 800;
        }

        .ui-section-title span {
            color: var(--accent-plum);
        }

        .ui-muted-action {
            color: var(--accent-plum);
            font-size: 0.92rem;
            font-weight: 700;
        }

        .upload-intro {
            margin-bottom: 0.85rem;
            padding: 1.2rem 1.2rem 0;
        }

        .upload-title {
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 800;
            color: #FFFFFF;
            margin: 0.55rem 0 0.35rem;
        }

        .upload-copy {
            color: #DCC9D4;
            margin: 0;
            max-width: 470px;
            line-height: 1.7;
        }

        .upload-icon {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-size: 1.45rem;
            color: #F6D8E3;
        }

        .support-copy {
            margin-top: 0.8rem;
            color: var(--text-soft);
            font-size: 0.9rem;
        }

        .meta-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.8rem;
            border-radius: var(--radius-pill);
            background: rgba(242, 184, 198, 0.16);
            color: var(--accent-plum);
            border: 1px solid rgba(217, 140, 166, 0.26);
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .notice {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin: 0.6rem 0 1rem;
            border: 1px solid rgba(217, 140, 166, 0.26);
            background: rgba(242, 184, 198, 0.17);
            color: var(--text-main);
            line-height: 1.55;
        }

        .notice.notice-error {
            background: rgba(155, 63, 104, 0.10);
            border-color: rgba(155, 63, 104, 0.20);
        }

        .recent-card {
            background: #FFFFFF;
            border-radius: 20px;
            border: 1px solid rgba(232, 222, 227, 0.95);
            box-shadow: 0 10px 24px rgba(36, 29, 43, 0.06);
            padding: 0.9rem;
            min-height: 240px;
        }

        .recent-thumb {
            position: relative;
            height: 112px;
            border-radius: 18px;
            background:
                radial-gradient(circle at 30% 20%, rgba(242, 184, 198, 0.18), transparent 28%),
                linear-gradient(160deg, #2A2232 0%, #1F1A27 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .recent-thumb::after {
            content: "";
            position: absolute;
            inset: auto -10% -20% -10%;
            height: 58px;
            background: radial-gradient(circle, rgba(217, 140, 166, 0.24) 0%, rgba(217, 140, 166, 0) 70%);
            filter: blur(10px);
        }

        .recent-play {
            width: 54px;
            height: 54px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.18);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            font-size: 1.1rem;
        }

        .recent-delete {
            position: absolute;
            top: 12px;
            right: 12px;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(217, 140, 166, 0.88);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
        }

        .recent-name {
            margin: 0.85rem 0 0.25rem;
            font-size: 1rem;
            font-weight: 800;
        }

        .recent-meta {
            margin: 0;
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .result-preview {
            border-radius: 24px;
            min-height: 220px;
            padding: 1.4rem;
            background:
                radial-gradient(circle at 20% 15%, rgba(242, 184, 198, 0.15), transparent 28%),
                radial-gradient(circle at 80% 80%, rgba(217, 140, 166, 0.14), transparent 30%),
                linear-gradient(180deg, #241D2B 0%, #1F1A27 100%);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }

        .result-preview::before,
        .result-preview::after {
            content: "";
            position: absolute;
            width: 160%;
            height: 70px;
            left: -30%;
            border-radius: 50%;
            border: 1px solid rgba(242, 184, 198, 0.12);
        }

        .result-preview::before {
            bottom: 30px;
        }

        .result-preview::after {
            bottom: 65px;
        }

        .result-word {
            position: relative;
            z-index: 1;
            font-size: clamp(1.7rem, 2.2vw, 2.55rem);
            font-weight: 800;
            line-height: 1.15;
            color: #F2B8C6;
            margin: 0;
        }

        .result-empty {
            position: relative;
            z-index: 1;
            color: #E5D8DF;
            font-size: 1rem;
            line-height: 1.8;
            max-width: 260px;
            margin: 0 auto;
        }

        .details-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }

        .details-block {
            padding: 1rem;
            border-radius: 20px;
            background: rgba(250, 247, 245, 0.9);
            border: 1px solid rgba(232, 222, 227, 0.8);
        }

        .details-label {
            margin: 0 0 0.35rem;
            color: var(--text-soft);
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }

        .details-value {
            margin: 0;
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 800;
        }

        .details-value.details-accent {
            color: var(--accent-plum);
        }

        .dashed-divider {
            border-top: 1.5px dashed rgba(217, 140, 166, 0.45);
            margin: 1rem 0;
        }

        .prob-row {
            display: grid;
            grid-template-columns: minmax(72px, 1fr) minmax(120px, 2fr) 60px;
            gap: 0.7rem;
            align-items: center;
            margin-bottom: 0.8rem;
        }

        .prob-word {
            font-weight: 700;
            color: var(--text-main);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .prob-bar {
            width: 100%;
            height: 12px;
            border-radius: var(--radius-pill);
            background: rgba(232, 222, 227, 0.9);
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: var(--radius-pill);
            background: linear-gradient(135deg, var(--accent-plum) 0%, var(--accent-pink) 100%);
        }

        .prob-value {
            text-align: right;
            font-weight: 800;
            color: var(--accent-plum);
        }

        @media (max-width: 991px) {
            .main .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .details-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
