import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation

from GestureRecognition.labeling import normalize_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.pkl"

SEQUENCES_PER_CLASS = None   # None = ALLE Aufnahmen pro Klasse überlagern (z. B. alle 30 für "A")


def _load_raw_tracks(label_dir: Path):
    """Laedt alle Rohtrajektorien aus einem Label-Ordner."""
    tracks = []
    for f in sorted(label_dir.glob("*.pickle")):
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        track = np.asarray(data["track"] if isinstance(data, dict) else data, dtype=float)
        if len(track) >= 8:
            tracks.append(track)
    return tracks


def visualize_dataset():
    """Zeigt Trajektorien des Datensatzes – mehrere Beispiele pro Klasse."""
    label_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if not label_dirs:
        print("Keine Daten gefunden in", RAW_DIR)
        return

    n = len(label_dirs)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, label_dir in enumerate(label_dirs):
        ax = axes[i]
        tracks = _load_raw_tracks(label_dir)
        label = label_dir.name

        ax.set_title(f"{label}  ({len(tracks)} Aufn.)")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        shown = tracks if SEQUENCES_PER_CLASS is None else tracks[:SEQUENCES_PER_CLASS]
        for track in shown:
            ax.plot(track[:, 0], track[:, 1], alpha=0.35, linewidth=1.0)
            ax.plot(track[0, 0], track[0, 1], "go", markersize=3)
            ax.plot(track[-1, 0], track[-1, 1], "rs", markersize=3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    anzahl = "alle" if SEQUENCES_PER_CLASS is None else f"max. {SEQUENCES_PER_CLASS}"
    plt.suptitle(f"Trajektorien pro Klasse  (gruen=Start, rot=Ende, {anzahl} Aufnahmen pro Klasse)")
    plt.tight_layout()
    plt.show()


def replay_recordings():
    """Spielt aufgenommene Sequenzen als Animation ab – eine pro Klasse."""
    label_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if not label_dirs:
        print("Keine Daten gefunden in", RAW_DIR)
        return

    print(f"Klassen: {[d.name for d in label_dirs]}")
    print("Fenster schliessen, um zur naechsten Klasse zu wechseln.\n")

    for label_dir in label_dirs:
        files = sorted(label_dir.glob("*.pickle"))
        if not files:
            continue

        path = files[0]
        with open(path, "rb") as f:
            data = pickle.load(f)
        track = np.asarray(data["track"] if isinstance(data, dict) else data, dtype=float)

        print(f"Replay: {path.name}  ({len(track)} Punkte)")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"Replay: {path.name}")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        line, = ax.plot([], [], "b-", linewidth=2)
        dot, = ax.plot([], [], "ro", markersize=8)

        def update(i, track=track):
            line.set_data(track[:i + 1, 0], track[:i + 1, 1])
            dot.set_data([track[i, 0]], [track[i, 1]])
            return line, dot

        _ani = FuncAnimation(fig, update, frames=len(track), interval=50, blit=True)
        plt.tight_layout()
        plt.show()


def evaluate_classifier():
    """Bewertet den HMMClassifier mit Accuracy und Confusion Matrix."""
    from GestureRecognition.hmmclassifier import HMMClassifier

    # Datensatz laden
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "rb") as f:
            dataset = pickle.load(f)
        print(f"Datensatz geladen: {DATASET_PATH.name}")
    else:
        print("Kein dataset.pkl gefunden – lade Rohdaten und normalisiere...")
        dataset = {}
        for d in sorted(RAW_DIR.iterdir()):
            if not d.is_dir():
                continue
            tracks = _load_raw_tracks(d)
            if tracks:
                dataset[d.name] = [normalize_trajectory(t) for t in tracks]

    if not dataset:
        print("Keine Daten gefunden.")
        return

    # 80/20 Train-Test-Split
    train_data, test_data = {}, {}
    for label, seqs in dataset.items():
        split = max(1, int(len(seqs) * 0.8))
        train_data[label] = seqs[:split]
        test_data[label] = seqs[split:]

    labels = sorted(dataset.keys())
    print(f"Klassen: {labels}")
    print(f"Training: {sum(len(v) for v in train_data.values())} Sequenzen")
    print(f"Test:     {sum(len(v) for v in test_data.values())} Sequenzen")

    # Classifier trainieren
    clf = HMMClassifier()
    clf.fit(train_data)

    # Vorhersagen
    y_true, y_pred = [], []
    for label in labels:
        for seq in test_data.get(label, []):
            pred = clf.predict([seq])[0]
            y_true.append(label)
            y_pred.append(pred)

    if not y_true:
        print("Zu wenige Testdaten (mind. 2 Aufnahmen pro Klasse noetig).")
        return

    # Accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    print(f"\nAccuracy: {accuracy:.2%}  ({correct}/{len(y_true)})")

    # Confusion Matrix
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if p in idx:
            cm[idx[t]][idx[p]] += 1

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Vorhersage")
    ax.set_ylabel("Tatsaechlich")
    ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy:.2%})")
    plt.colorbar(im, ax=ax)

    threshold = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualisierung & Evaluation des Gesten-Datensatzes."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="dataset",
        choices=["dataset", "replay", "evaluate"],
        help="dataset = Trajektorien pro Klasse, "
             "replay = Aufnahmen abspielen, "
             "evaluate = Accuracy + Confusion Matrix (braucht HMMClassifier). "
             "Standard: dataset",
    )
    args = parser.parse_args()

    if args.command == "dataset":
        visualize_dataset()
    elif args.command == "replay":
        replay_recordings()
    elif args.command == "evaluate":
        evaluate_classifier()
