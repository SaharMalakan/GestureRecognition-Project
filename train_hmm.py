"""
Trainiert den HMMClassifier auf dem gebauten Datensatz und speichert das Modell.

Ablauf:
    1. python -m GestureRecognition.labeling ...      # Aufnahmen machen
    2. python -c "from GestureRecognition.labeling import dataset_building; dataset_building('data/dataset.pkl')"
    3. python train_hmm.py                            # <-- dieses Skript: trainiert & speichert data/hmm.pkl

Danach liegt das Modell unter data/hmm.pkl, wo der Live-Demo (HMMModule) es erwartet.
"""
import pickle
from pathlib import Path

from GestureRecognition.hmmclassifier import HMMClassifier

DATASET_PATH = Path("data/dataset.pkl")  # Eingabe: fertiger Datensatz
MODEL_PATH = Path("data/hmm.pkl")        # Ausgabe: trainiertes Modell


def main():
    # Abbruch, falls der Datensatz noch nicht gebaut wurde
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"'{DATASET_PATH}' fehlt. Erst den Datensatz bauen:\n"
            "  python -c \"from GestureRecognition.labeling import dataset_building; "
            "dataset_building('data/dataset.pkl')\""
        )

    # Datensatz von der Platte laden
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    gesamt = sum(len(v) for v in dataset.values())
    print(f"Datensatz geladen: {len(dataset)} Klassen, {gesamt} Sequenzen")

    # Training: ein HMM pro Klasse (siehe hmmclassifier.py)
    clf = HMMClassifier()
    clf.fit(dataset)
    clf.save(MODEL_PATH)  # als hmm.pkl speichern

    print(f"Modell gespeichert: {MODEL_PATH}  (Klassen: {', '.join(clf.labels)})")


if __name__ == "__main__":
    main()
