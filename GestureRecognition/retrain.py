"""
Retraining in einem Befehl: dataset bauen -> trainieren -> speichern.

Macht genau diese drei Schritte:
    1. Alle Aufnahmen aus data/raw/ einlesen und normalisieren
    2. HMMClassifier auf dem frischen Datensatz trainieren
    3. Fertiges Modell als data/hmm.pkl speichern (HMMModule lädt es dann)

Aufruf über die Konsole:
    python -m GestureRecognition.retrain
"""

from pathlib import Path

from GestureRecognition.hmmclassifier import HMMClassifier
from GestureRecognition.labeling import dataset_building, PROJECT_ROOT

DATASET_PATH = PROJECT_ROOT / "data" / "dataset.pkl"
MODEL_PATH   = PROJECT_ROOT / "data" / "hmm.pkl"


def retrain(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    """Baut den Datensatz neu, trainiert den Classifier und speichert ihn."""

    # Schritt 1: alle Rohdaten aus data/raw/ laden und normalisieren
    print("=== Schritt 1: Datensatz einlesen ===")
    dataset = dataset_building(dataset_path)

    # Schritt 2: für jede Klasse ein eigenes HMM trainieren
    print("\n=== Schritt 2: HMM trainieren ===")
    classifier = HMMClassifier()
    classifier.fit(dataset)
    print(f"Klassen gelernt: {classifier.labels}")

    # Schritt 3: fertig trainiertes Modell speichern,
    # damit HMMModule (hiddenmarkov.py) es beim nächsten Start laden kann
    print("\n=== Schritt 3: Modell speichern ===")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_path)
    print(f"Modell gespeichert: {model_path}")

    return classifier


if __name__ == "__main__":
    retrain()
