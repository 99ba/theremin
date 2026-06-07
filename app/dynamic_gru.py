from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dynamic_gesture import DYNAMIC_TEMPLATES_PATH, _resample_sequence
from .music_binding import normalise_template_binding

GRU_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "dynamic_gesture_gru.pt"


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError:
        return None, None, None
    return torch, nn, functional


def _load_sequences(path: Path, sequence_length: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if not path.exists():
        return np.zeros((0, sequence_length, 15), dtype=np.float32), np.zeros(0, dtype=np.int64), []

    with open(path, "r", encoding="utf-8") as fh:
        gestures = json.load(fh).get("gestures", [])

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    labels: list[dict] = []

    for gesture in gestures:
        try:
            binding = normalise_template_binding(gesture)
        except (KeyError, TypeError, ValueError):
            continue
        label_index = len(labels)
        labels.append(
            {
                "name": gesture.get("gesture_name") or gesture["name"],
                "binding_type": binding.binding_type,
                "binding_name": binding.binding_name,
                "midi_notes": [int(note) for note in binding.midi_notes],
                "note_name": binding.binding_name,
                "midi": int(binding.midi_notes[0]),
            }
        )
        for raw_seq in gesture.get("sequences", []):
            seq = np.asarray(raw_seq, dtype=np.float32)
            if seq.ndim != 2 or len(seq) < 2:
                continue
            resampled = _resample_sequence([row for row in seq], sequence_length)
            if resampled is None:
                continue
            x_rows.append(resampled)
            y_rows.append(label_index)

    if not x_rows:
        return np.zeros((0, sequence_length, 15), dtype=np.float32), np.zeros(0, dtype=np.int64), labels
    return np.stack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int64), labels


def _make_model(nn, input_dim: int, hidden_dim: int, num_classes: int):
    class DynamicGestureGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])

    return DynamicGestureGRU()


def train_dynamic_gru_from_templates(
    templates_path: Path = DYNAMIC_TEMPLATES_PATH,
    model_path: Path = GRU_MODEL_PATH,
    sequence_length: int = 60,
    epochs: int = 120,
) -> bool:
    torch, nn, functional = _try_import_torch()
    if torch is None or nn is None or functional is None:
        print("  PyTorch is not installed; dynamic GRU training skipped, using sequence template matching.")
        return False

    x_np, y_np, labels = _load_sequences(Path(templates_path), sequence_length)
    if len(labels) < 2 or len(x_np) < 2:
        print("  Dynamic GRU needs at least two gesture classes / sequences; using sequence template matching.")
        return False

    model = _make_model(nn, input_dim=x_np.shape[-1], hidden_dim=48, num_classes=len(labels))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    model.train()
    for _ in range(max(int(epochs), 1)):
        logits = model(x)
        loss = functional.cross_entropy(logits, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "sequence_length": int(sequence_length),
            "input_dim": int(x_np.shape[-1]),
            "hidden_dim": 48,
        },
        model_path,
    )
    print(f"  Trained dynamic GRU model: {model_path}")
    return True


class DynamicGRUPredictor:
    def __init__(self, model_path: Path = GRU_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.torch = None
        self.functional = None
        self.model = None
        self.labels: list[dict] = []
        self.sequence_length = 60

    def load(self) -> bool:
        torch, nn, functional = _try_import_torch()
        if torch is None or nn is None or functional is None or not self.model_path.exists():
            return False
        try:
            payload = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(self.model_path, map_location="cpu")
        self.sequence_length = int(payload["sequence_length"])
        self.labels = list(payload["labels"])
        self.model = _make_model(
            nn,
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_classes=len(self.labels),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.torch = torch
        self.functional = functional
        return True

    def predict(self, sequence: np.ndarray) -> tuple[str | None, str | None, list[int] | None, str | None, float]:
        if self.model is None or self.torch is None or self.functional is None or not self.labels:
            return None, None, None, None, 0.0
        seq = _resample_sequence([row for row in sequence], self.sequence_length)
        if seq is None:
            return None, None, None, None, 0.0
        with self.torch.no_grad():
            x = self.torch.tensor(seq[None, :, :], dtype=self.torch.float32)
            probs = self.functional.softmax(self.model(x), dim=1)[0]
            confidence, index = self.torch.max(probs, dim=0)
        label = self.labels[int(index.item())]
        midi_notes = label.get("midi_notes") or [int(label["midi"])]
        return (
            label["name"],
            label.get("binding_name") or label.get("note_name"),
            [int(note) for note in midi_notes],
            label.get("binding_type", "note"),
            float(confidence.item()),
        )
