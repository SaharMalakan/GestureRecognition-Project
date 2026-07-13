"""
TEMP-DEBUG — spurlos loeschbar.

Bildet den LIVE-Pfad der Demo (main.py) nach, um zu sehen, WAS beim HMM
ankommt: dieselbe Kamera-Aufbereitung (resize 640x360 + flip), derselbe
Gesture-Recognizer, dieselbe Steuerung (Zeigefinger=Start, Faust=Stopp),
derselbe Finger (Index 8), dieselbe Normalisierung (normalize_trajectory)
und dasselbe Modell (data/hmm.pkl).

Nach jedem gemalten Buchstaben:
  - Vorhersage auf der Spur, ihrer SPIEGELUNG und ihrer UMKEHRUNG (Richtung)
  - ein Vergleichsbild  debug_live_<n>.png  (deine Spur vs. Trainings-Referenz)
  - alles zusaetzlich in  debug_live.pkl  gespeichert (fuer Detailanalyse)

Aufruf:   python debug_live.py
Steuerung: Zeigefinger hoch = Start, Faust = Stopp, Taste q/ESC = beenden.

SPURLOS LOESCHEN:
  rm debug_live.py debug_live.pkl debug_live_*.png     (Git Bash)
  del debug_live.py debug_live.pkl debug_live_*.png     (cmd)
Es wurde KEINE andere Datei veraendert.
"""
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GestureRecognition.labeling import (
    normalize_trajectory, _webcam_settings, GESTURE_MODEL, FINGER_IDX,
)

TRIGGER_FRAMES = 3
START_GESTURE = "Pointing_Up"
STOP_GESTURE = "Closed_Fist"


def load_model():
    with open("data/hmm.pkl", "rb") as f:
        return pickle.load(f)


def load_reference():
    """Ein normalisiertes Trainings-Beispiel je Klasse (zum Vergleich)."""
    try:
        with open("data/dataset.pkl", "rb") as f:
            ds = pickle.load(f)
        return {lbl: np.asarray(seqs[0], float) for lbl, seqs in ds.items() if seqs}
    except Exception:
        return {}


def predict_all(model, norm):
    """Vorhersage auf Spur, Spiegelung (links<->rechts) und Umkehrung (Richtung)."""
    mirror = norm.copy(); mirror[:, 0] = -mirror[:, 0]
    reverse = norm[::-1].copy()
    def top(seq):
        lbl = model.predict([seq])[0]
        sc = float(np.max(model.decision_function([seq])[0]))
        return lbl, sc
    return {"normal": top(norm), "gespiegelt": top(mirror), "umgekehrt": top(reverse)}


def save_plot(idx, norm, pred_label, ref):
    fig, ax = plt.subplots(figsize=(4, 4))
    if pred_label in ref:
        r = ref[pred_label]
        ax.plot(r[:, 0], r[:, 1], color="0.75", lw=4, label=f"Training '{pred_label}'")
    ax.plot(norm[:, 0], norm[:, 1], "b-", lw=2, label="deine Live-Spur")
    ax.plot(norm[0, 0], norm[0, 1], "go", ms=9)   # Start gruen
    ax.plot(norm[-1, 0], norm[-1, 1], "rs", ms=9)  # Ende rot
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(fontsize=8)
    ax.set_title(f"#{idx}: erkannt als '{pred_label}'  (gruen=Start, rot=Ende)")
    out = f"debug_live_{idx}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    model = load_model()
    ref = load_reference()
    print("Modell geladen. Klassen:", ", ".join(model.labels))

    with open(GESTURE_MODEL, "rb") as f:
        recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_buffer=f.read()),
                num_hands=1,
            )
        )

    cam = _webcam_settings()
    cap = cv2.VideoCapture(cam["device"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    if not cap.isOpened():
        print("Webcam konnte nicht geoeffnet werden (deviceIndex in config.yml pruefen).")
        return

    state = "idle"
    track = []
    last_gesture, streak = None, 0
    saved = []
    idx = 0

    print("\nZeigefinger hoch = Start, Faust = Stopp, q = beenden.\n")

    # Kleines, frei skalierbares Fenster oben links (sonst schneidet der Rand
    # den unteren Text ab).
    WIN = "DEBUG live (q = beenden)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 480, 270)
    cv2.moveWindow(WIN, 20, 20)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            # EXAKT wie die Demo-Webcam: erst auf Zielgroesse, dann flip
            frame = cv2.resize(frame, (cam["width"], cam["height"]))
            if cam["flip"]:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            gesture = result.gestures[0][0].category_name if result.gestures else "None"
            fingertip = None
            if result.hand_landmarks:
                lm = result.hand_landmarks[0][FINGER_IDX]
                fingertip = (lm.x, lm.y)

            streak = streak + 1 if gesture == last_gesture else 1
            last_gesture = gesture
            stable = streak >= TRIGGER_FRAMES

            if state == "idle":
                if gesture == START_GESTURE and stable:
                    state, track = "recording", []
            elif state == "recording":
                if fingertip is not None:
                    track.append(fingertip)
                if gesture == STOP_GESTURE and stable:
                    state = "idle"
                    if len(track) >= 8:
                        idx += 1
                        raw = np.array(track, float)
                        norm = normalize_trajectory(raw)
                        res = predict_all(model, norm)
                        png = save_plot(idx, norm, res["normal"][0], ref)
                        saved.append({"raw": raw, "norm": norm, "pred": res})
                        with open("debug_live.pkl", "wb") as f:
                            pickle.dump(saved, f)
                        print(f"#{idx}  ({len(track)} Punkte)")
                        for k, (lbl, sc) in res.items():
                            print(f"    {k:11s}: {lbl}   (score {sc:.1f})")
                        print(f"    Bild: {png}\n")
                    else:
                        print("  (zu kurz, verworfen)")
                    track = []

            # kleine Anzeige
            pts = [(int(x * w), int(y * h)) for (x, y) in track]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)
            txt = "Zeigefinger HOCH = Start" if state == "idle" else f"malen... ({len(track)}) -> FAUST = Stop"
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 200, 200) if state == "idle" else (0, 0, 255), 2)
            cv2.putText(frame, f"Geste: {gesture}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("DEBUG live (q = beenden)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        print(f"\nFertig. {idx} Buchstaben gespeichert in debug_live.pkl + debug_live_*.png")


if __name__ == "__main__":
    main()
