from __future__ import annotations

import importlib
from pathlib import Path

from UI.models import PredictionResult, build_prediction_result


class PredictionServiceUnavailableError(RuntimeError):
    """Raised when the prediction backend has not been wired up yet."""


class PredictionService:
    """Thin integration layer between the UI and any prediction backend."""

    def predict(self, video_path: str | Path, model_id: str) -> PredictionResult:
        backend = self._load_backend()
        response = backend.predict_word_level(Path(video_path), model_id)
        return self._normalize_response(response)

    def _load_backend(self):
        try:
            return importlib.import_module("UI.prediction_backend")
        except ModuleNotFoundError as exc:
            raise PredictionServiceUnavailableError(
                "Prediction backend is not configured yet. "
                "Add UI/prediction_backend.py with a predict_word_level(video_path, model_id) function."
            ) from exc

    @staticmethod
    def _normalize_response(response: dict) -> PredictionResult:
        if not isinstance(response, dict):
            raise ValueError("Prediction backend must return a dictionary.")

        predicted_word = str(response.get("predictedWord") or response.get("predicted_word") or "").strip()
        confidence = response.get("confidence")
        probabilities = response.get("probabilities", [])

        if not predicted_word:
            raise ValueError("Prediction response is missing predictedWord.")
        if confidence is None:
            raise ValueError("Prediction response is missing confidence.")

        normalized_probabilities: list[tuple[str, float | int]] = []
        for item in probabilities:
            if isinstance(item, dict):
                label = item.get("label")
                probability = item.get("probability")
            else:
                label = probability = None
            if label is None or probability is None:
                continue
            normalized_probabilities.append((str(label), probability))

        return build_prediction_result(predicted_word, confidence, normalized_probabilities)
