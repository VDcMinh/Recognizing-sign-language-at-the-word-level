from __future__ import annotations

from pathlib import Path
import tkinter as tk

from UI.components import BG, configure_ttk_styles
from UI.home import HomePage
from UI.models import ModelOption
from UI.services import PredictionService


def build_app() -> tk.Tk:
    root = tk.Tk()
    root.title("Sign Language Prediction (Word Level)")
    root.configure(bg=BG)
    root.geometry("1440x1080")
    root.minsize(420, 860)
    configure_ttk_styles(root)

    project_root = Path(__file__).resolve().parent.parent
    assets_dir = project_root / "Elements"
    sample_videos_dir = project_root / "Test Videos"
    models = [
        ModelOption(id="word-level-transformer", label="Word Level Transformer"),
        ModelOption(id="best-pt-checkpoint", label="best.pt Checkpoint"),
        ModelOption(id="lightweight-baseline", label="Lightweight Baseline"),
    ]

    page = HomePage(
        root,
        assets_dir=assets_dir,
        sample_videos_dir=sample_videos_dir,
        models=models,
        prediction_service=PredictionService(),
    )
    page.pack(fill="both", expand=True)
    return root


def main() -> None:
    app = build_app()
    app.mainloop()


if __name__ == "__main__":
    main()
