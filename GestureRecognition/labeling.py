"""
Datenerfassung und Datensatz-Aufbau für die Gestenerkennung.

Idee:
-----
- ``data_labeling`` nimmt Buchstaben/Gesten auf. Gesteuert wird das über
  Handgesten (MediaPipe Gesture Recognizer):
      * Zeigefinger nach oben (Pointing_Up)  -> Aufnahme START
      * Faust (Closed_Fist)                  -> Aufnahme STOPP
      * Taste 's'                            -> Aufnahme speichern
  Während der Aufnahme wird die Spur der Zeigefinger-Spitze gesammelt.
  Jede gespeicherte Aufnahme landet als ``.pickle`` in ``data/raw/<label>/``.
- ``dataset_building`` lädt alle Aufnahmen wieder, normalisiert die Spur
  (genau wie der Preprocessor live) und baut daraus einen Datensatz fürs HMM.

Aufnehmen geht am einfachsten direkt über die Konsole:

    python -m GestureRecognition.labeling <label> [anzahl]
    # z.B.:  python -m GestureRecognition.labeling A 10

Hinweis: dafür wird das Modell ``gesture_recognizer.task`` im Projekt-Root
gebraucht (siehe Fehlermeldung beim Start, falls es fehlt).
"""

import sys
import pickle
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- Pfade / Einstellungen ---------------------------------------------------

# labeling.py liegt in GestureRecognition/, der Projekt-Root ist eine Ebene höher.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"                 # Roh-Aufnahmen je Label
GESTURE_MODEL = PROJECT_ROOT / "gesture_recognizer.task"  # MediaPipe Gesture Recognizer

FINGER_IDX = 8          # Zeigefinger-Spitze (gleicher Wert wie im Preprocessor)
MIN_SEQUENCE_LEN = 8    # kürzere Aufnahmen werfen wir beim Datensatz-Bau raus

# Steuer-Gesten (Namen kommen so vom MediaPipe Gesture Recognizer)
START_GESTURE = "Pointing_Up"
STOP_GESTURE = "Closed_Fist"
TRIGGER_FRAMES = 3      # so viele Frames muss eine Geste halten, damit sie zählt (gegen Flackern)


# --- Aufnahme ----------------------------------------------------------------

def data_labeling(times: int, label: str):
    """
    Nimmt ``times`` Gesten/Buchstaben für die Klasse ``label`` auf.

    Ablauf pro Aufnahme:
      1. Zeigefinger nach oben halten  -> Aufnahme startet.
      2. Buchstaben in die Luft "malen" (die Fingerspur wird gesammelt).
      3. Faust machen                  -> Aufnahme stoppt.
      4. 's' speichern / 'v' verwerfen / 'q' beenden.

    Gespeichert wird nach ``data/raw/<label>/<label>_NNN.pickle``.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen behalten werden sollen.
    label : str
        Name der Geste / des Buchstabens (z.B. "A", "B", "kreis").
    """
    label_dir = RAW_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    if not GESTURE_MODEL.exists():
        raise FileNotFoundError(
            f"Modelldatei '{GESTURE_MODEL.name}' nicht gefunden. "
            "Lade sie herunter von: https://storage.googleapis.com/"
            "mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/"
            "gesture_recognizer.task  (in den Projekt-Root legen)"
        )

    # Gesture Recognizer laden. Der liefert pro Frame die erkannte Geste UND
    # die Hand-Landmarks - wir brauchen also nur dieses eine Modell.
    # Modell als Bytes einlesen statt per Pfad zu uebergeben: MediaPipes interner
    # Lader kommt mit Sonderzeichen im Pfad (z.B. das "ue" in "Duesseldorf") nicht
    # klar und wirft sonst einen FileNotFoundError, obwohl die Datei da ist.
    with open(GESTURE_MODEL, "rb") as f:
        model_bytes = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_bytes)
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=1)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cam = _webcam_settings()
    cap = cv2.VideoCapture(cam["device"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    if not cap.isOpened():
        recognizer.close()
        raise RuntimeError(f"Webcam {cam['device']} konnte nicht geöffnet werden "
                           "(deviceIndex in config.yml prüfen).")

    window = f"Labeling: {label}"
    state = "idle"              # idle -> recording -> review
    track = []                  # gesammelte Fingerpositionen der aktuellen Aufnahme
    last_gesture, streak = None, 0
    saved = 0

    try:
        while saved < times:
            ok, frame = cap.read()
            if not ok:
                continue
            if cam["flip"]:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe erwartet RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_image)

            # erkannte Geste (bestes Ergebnis der ersten Hand)
            gesture = result.gestures[0][0].category_name if result.gestures else "None"

            # Position der Zeigefinger-Spitze (falls eine Hand erkannt wurde)
            fingertip = None
            if result.hand_landmarks:
                lm = result.hand_landmarks[0][FINGER_IDX]
                fingertip = (lm.x, lm.y)

            # Geste entprellen: erst zählen, wenn sie ein paar Frames stabil ist
            streak = streak + 1 if gesture == last_gesture else 1
            last_gesture = gesture
            stable = streak >= TRIGGER_FRAMES

            # --- Zustandslogik ---
            if state == "idle":
                if gesture == START_GESTURE and stable:
                    state = "recording"
                    track = []
            elif state == "recording":
                if fingertip is not None:
                    track.append(fingertip)
                if gesture == STOP_GESTURE and stable:
                    # die letzten Frames sind schon die Faust -> abschneiden
                    track = track[:-TRIGGER_FRAMES] if len(track) > TRIGGER_FRAMES else []
                    state = "review"

            _draw_overlay(frame, state, label, saved, times, track, gesture)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:   # q oder ESC
                break

            # im Review-Zustand auf Tastendruck warten
            if state == "review":
                if key == ord("s"):
                    if len(track) < MIN_SEQUENCE_LEN:
                        print(f"-> zu kurz ({len(track)} Punkte), nicht gespeichert.")
                    else:
                        idx = _next_index(label_dir, label)
                        target = label_dir / f"{label}_{idx:03d}.pickle"
                        with open(target, "wb") as f:
                            pickle.dump({"label": label,
                                         "finger_idx": FINGER_IDX,
                                         "track": np.array(track, dtype=float)}, f)
                        saved += 1
                        print(f"-> gespeichert: {target.name}  ({len(track)} Punkte)")
                    state, track = "idle", []
                elif key in (ord("v"), ord("n")):
                    print("-> verworfen.")
                    state, track = "idle", []
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()

    print(f"\nFertig. {saved} neue Aufnahme(n) für '{label}'.")


def _draw_overlay(frame, state, label, saved, times, track, gesture):
    """Zeichnet Fingerspur + Hinweistext ins Kamerabild (nur Anzeige)."""
    h, w = frame.shape[:2]

    # Fingerspur: normalisierte Koordinaten (0..1) -> Pixel
    pts = [(int(x * w), int(y * h)) for (x, y) in track]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)
    if pts:
        cv2.circle(frame, pts[-1], 6, (0, 255, 255), -1)

    # Hinweistext je Zustand (ASCII, weil OpenCV keine Umlaute kann)
    texte = {
        "idle":      f"'{label}'  [{saved}/{times}]  -> Zeigefinger HOCH = Start",
        "recording": f"Aufnahme laeuft ({len(track)})  -> FAUST = Stop",
        "review":    "[s] speichern   [v] verwerfen   [q] beenden",
    }
    farben = {"idle": (200, 200, 200), "recording": (0, 0, 255), "review": (0, 255, 0)}
    cv2.putText(frame, texte[state], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, farben[state], 2)
    cv2.putText(frame, f"Geste: {gesture}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def _webcam_settings():
    """Liest Webcam-Einstellungen aus config.yml (mit sinnvollen Defaults)."""
    cfg = {}
    try:
        import yaml
        with open(PROJECT_ROOT / "config.yml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass  # keine config -> Defaults
    wc = cfg.get("webcam", {}) if isinstance(cfg, dict) else {}
    return {
        "device": wc.get("deviceIndex", 0),
        "width": wc.get("width", 640),
        "height": wc.get("height", 360),
        "flip": wc.get("flip", True),
    }


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
    """Liest die rohe (T, 2)-Fingerspur aus einer Aufnahme."""
    with open(recording_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "track" in data:
        return np.asarray(data["track"], dtype=float)
    return np.asarray(data, dtype=float)  # Fallback für andere Formate


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
    """Kleiner Helfer zum Reinschauen, wie eine .pickle-Aufnahme aussieht."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        track = np.asarray(data.get("track", []))
        print(f"label={data.get('label')}  finger_idx={data.get('finger_idx')}  "
              f"punkte={len(track)}  shape={track.shape}")
    else:
        print(f"Typ: {type(data).__name__}")


# --- direkt aus der Konsole aufrufbar ---------------------------------------

if __name__ == "__main__":
    # Nutzung:  python -m GestureRecognition.labeling <label> [anzahl]
    if len(sys.argv) >= 2:
        gesten_label = sys.argv[1]
        anzahl = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        data_labeling(anzahl, gesten_label)
    else:
        print("Aufruf:  python -m GestureRecognition.labeling <label> [anzahl]")
        print("Beispiel: python -m GestureRecognition.labeling A 10")
