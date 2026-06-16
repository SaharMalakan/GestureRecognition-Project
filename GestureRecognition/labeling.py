"""
Datenerfassung und Datensatz-Aufbau für die Gestenerkennung.

Idee:
-----
- ``data_labeling`` nimmt mit der bestehenden Pipeline (HandDetector ->
  Preprocessor) Gesten auf. Dafür wird SignalHub als Subprocess im
  ``record``-Modus gestartet. Jede Aufnahme landet als ``.pickle`` in
  ``data/raw/<label>/``.
- ``dataset_building`` lädt alle Aufnahmen wieder, zieht pro Aufnahme die
  Fingerspur heraus, normalisiert sie (genau wie der Preprocessor live) und
  baut daraus einen Datensatz, mit dem der HMM-Klassifikator arbeiten kann.

Aufnehmen geht am einfachsten direkt über die Konsole:

    python -m GestureRecognition.labeling <label> [anzahl]
    # z.B.:  python -m GestureRecognition.labeling kreis 10
"""

import sys
import shutil
import pickle
import subprocess
from pathlib import Path

import numpy as np


# --- Pfade / Einstellungen ---------------------------------------------------

# labeling.py liegt in GestureRecognition/, der Projekt-Root ist eine Ebene höher.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"

RAW_DIR = PROJECT_ROOT / "data" / "raw"          # hier liegen die Roh-Aufnahmen je Label
TMP_FILE = PROJECT_ROOT / "record" / "_tmp.pickle"  # Zwischenablage für eine Aufnahme

FINGER_IDX = 8          # Zeigefinger-Spitze (gleicher Wert wie im Preprocessor)
MIN_SEQUENCE_LEN = 8    # kürzere Aufnahmen werfen wir beim Datensatz-Bau raus


# --- Tastatur-Eingabe (ein Tastendruck, ohne Enter) --------------------------

try:
    import msvcrt  # Windows

    def _getch():
        ch = msvcrt.getch()
        if ch in (b"\r", b"\n"):
            return "s"          # Enter = speichern
        if ch == b"\x1b":
            return "q"          # ESC   = beenden
        return ch.decode(errors="ignore").lower()

except ImportError:  # Fallback (Linux/Mac): einfach Enter drücken
    def _getch():
        antwort = input().strip().lower()
        return antwort[0] if antwort else "s"


# --- Aufnahme ----------------------------------------------------------------

def data_labeling(times: int, label: str):
    """
    Nimmt ``times`` Gesten für die Klasse ``label`` auf.

    Ablauf pro Aufnahme:
      1. SignalHub im record-Modus als Subprocess starten -> ein Fenster geht auf.
      2. Geste ausführen, dann das Fenster schließen (q / ESC).
      3. Aufnahme behalten oder verwerfen (Tastendruck).

    Behaltene Aufnahmen landen in ``data/raw/<label>/<label>_NNN.pickle``.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen behalten werden sollen.
    label : str
        Name der Geste / Klasse (z.B. "kreis", "links", "rechts").
    """
    label_dir = RAW_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    TMP_FILE.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    while saved < times:
        print(f"\n=== '{label}': Aufnahme {saved + 1}/{times} ===")
        print("Fenster geht auf -> Geste ausführen -> Fenster schließen (q / ESC).")

        # alte Zwischenablage entfernen, damit wir keine alte Aufnahme behalten
        if TMP_FILE.exists():
            TMP_FILE.unlink()

        # die normale Pipeline starten, nur eben aufnehmend.
        # sys.executable = der gleiche Python, mit dem wir hier laufen (gleiche Umgebung).
        result = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT),
             "--mode", "record",
             "--recorder.file", str(TMP_FILE)],
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0 or not TMP_FILE.exists():
            print("-> Keine Aufnahme entstanden. Nochmal versuchen.")
            continue

        print("[Enter/s] speichern   [v] verwerfen & nochmal   [q] beenden")
        key = _getch()

        if key == "q":
            print("Beendet.")
            break
        if key == "v":
            TMP_FILE.unlink()
            print("-> verworfen.")
            continue

        # behalten: in den Label-Ordner verschieben, fortlaufend nummeriert
        idx = _next_index(label_dir, label)
        target = label_dir / f"{label}_{idx:03d}.pickle"
        shutil.move(str(TMP_FILE), str(target))
        saved += 1
        print(f"-> gespeichert: {target.name}")

    print(f"\nFertig. {saved} neue Aufnahme(n) für '{label}'.")


def _next_index(label_dir: Path, label: str) -> int:
    """Nächste freie Nummer für ein Label finden (keine Aufnahme überschreiben)."""
    nums = []
    for p in label_dir.glob(f"{label}_*.pickle"):
        stamm = p.stem.replace(f"{label}_", "")
        if stamm.isdigit():
            nums.append(int(stamm))
    return max(nums) + 1 if nums else 0


# --- Datensatz bauen ---------------------------------------------------------

def normalize_trajectory(points):
    """
    Trajektorie normalisieren - genau wie im Preprocessor (modules/preprocessor.py).

    Zentrum abziehen und auf [-1, 1] skalieren. Dadurch wird die Geste
    unabhängig davon, *wo* im Bild und *wie groß* sie ausgeführt wurde.
    Wichtig: Training und Live-Erkennung müssen exakt gleich verarbeiten!
    """
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    if scale > 0:
        points = points / scale
    return points


def _extract_finger_track(recording_path: Path, finger_idx: int = FINGER_IDX):
    """
    Holt aus einer Aufnahme die Fingerspur als (T, 2)-Array.

    Eine Aufnahme ist ein dict: Modulname -> Liste der Frame-Ausgaben.
    Wir gehen über die ``detector``-Ausgaben und lesen pro Frame die
    Landmarke ``finger_idx`` (Zeigefinger-Spitze).
    """
    with open(recording_path, "rb") as f:
        ledger = pickle.load(f)

    detector_frames = ledger.get("detector", []) if isinstance(ledger, dict) else []

    track = []
    for frame in detector_frames:
        # je Frame steht das Ergebnis unter dem Signal-Namen "detector"
        result = frame.get("detector") if isinstance(frame, dict) else frame
        if result is None or not getattr(result, "hand_landmarks", None):
            continue  # in diesem Frame wurde keine Hand erkannt
        lm = result.hand_landmarks[0][finger_idx]
        track.append((lm.x, lm.y))

    return np.array(track, dtype=float)


def dataset_building(output_path):
    """
    Baut aus allen Aufnahmen in ``data/raw/`` einen Datensatz.

    Ergebnis-Format (einfach und gut weiterverarbeitbar):

        { "<label>": [ seq0, seq1, ... ], ... }

    wobei jede ``seq`` ein normalisiertes (T, 2)-numpy-Array ist (eine Geste).
    Daraus kann der HMMClassifier später pro Klasse seine Sequenzen ziehen.

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Datensatz (z.B. "data/dataset.pkl").
    """
    dataset = {}

    for label_dir in sorted(RAW_DIR.glob("*")):
        if not label_dir.is_dir():
            continue
        label = label_dir.name

        sequences = []
        for rec in sorted(label_dir.glob("*.pickle")):
            track = _extract_finger_track(rec)
            if len(track) < MIN_SEQUENCE_LEN:
                print(f"  übersprungen (nur {len(track)} Punkte): {rec.name}")
                continue
            sequences.append(normalize_trajectory(track))

        if sequences:
            dataset[label] = sequences
            print(f"{label}: {len(sequences)} Sequenzen")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    klassen = len(dataset)
    gesamt = sum(len(v) for v in dataset.values())
    print(f"\nDatensatz gespeichert: {output_path}  ({klassen} Klassen, {gesamt} Sequenzen)")
    return dataset


def inspect_recording(path):
    """
    Kleiner Helfer zum Reinschauen: zeigt, wie eine .pickle-Aufnahme aufgebaut ist.

    Praktisch, um einmal zu prüfen, unter welchen Keys die Signale liegen,
    falls die Extraktion in ``_extract_finger_track`` angepasst werden muss.
    """
    with open(path, "rb") as f:
        ledger = pickle.load(f)

    if not isinstance(ledger, dict):
        print(f"Typ: {type(ledger).__name__}")
        return

    print(f"Module/Signale in der Aufnahme: {list(ledger.keys())}")
    for name, frames in ledger.items():
        n = len(frames) if hasattr(frames, "__len__") else "?"
        beispiel = ""
        if hasattr(frames, "__len__") and len(frames) > 0:
            f0 = frames[0]
            beispiel = f" | Frame-Keys: {list(f0.keys())}" if isinstance(f0, dict) else f" | {type(f0).__name__}"
        print(f"  {name}: {n} Frames{beispiel}")


# --- direkt aus der Konsole aufrufbar ---------------------------------------

if __name__ == "__main__":
    # Nutzung:  python -m GestureRecognition.labeling <label> [anzahl]
    if len(sys.argv) >= 2:
        gesten_label = sys.argv[1]
        anzahl = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        data_labeling(anzahl, gesten_label)
    else:
        print("Aufruf:  python -m GestureRecognition.labeling <label> [anzahl]")
        print("Beispiel: python -m GestureRecognition.labeling kreis 10")
