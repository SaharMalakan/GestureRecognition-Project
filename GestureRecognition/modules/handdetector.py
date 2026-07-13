import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from SignalHub import GALY, bgr, Module

mp_hand = mp.tasks.vision.HandLandmarksConnections


def draw_hand_landmarks(hand_landmarks, galy: GALY):
    """Zeichnet die Knochen (Linien) und Gelenke (Kreise) einer Hand.

    ``hand_landmarks`` ist die Landmarken-Liste *einer* Hand aus MediaPipe.
    Jede Landmarke hat normalisierte Koordinaten ``.x`` und ``.y`` (Werte 0..1).
    Pro Finger werden zuerst die Verbindungslinien und danach die Punkte gezeichnet.
    """
    lm = {
        "thumb":         {"color": bgr("#0000FF")},
        "index_finger":  {"color": bgr("#00FF00")},
        "middle_finger": {"color": bgr("#FF0000")},
        "ring_finger":   {"color": bgr("#00FFFF")},
        "pinky_finger":  {"color": bgr("#FF00FF")},
        "palm":          {"color": bgr("#C8C8C8")},
    }
    for key in lm.keys():
        pts = set()
        for conn in getattr(mp_hand, f"HAND_{key.upper()}_CONNECTIONS"):
            start = (hand_landmarks[conn.start].x,
                    hand_landmarks[conn.start].y)
            end = (hand_landmarks[conn.end].x,
                hand_landmarks[conn.end].y)
            galy.line(start, end, lm[key]["color"], 2)
            pts.update([conn.start, conn.end])
        for pt in pts:
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 5, (255,255,255), 1)
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 4, lm[key]["color"], -1)


class HandDetector(Module):
    """Erkennt Hände im Kamerabild und liefert Landmarken + Geste.

    Das Modul nutzt den MediaPipe *Gesture Recognizer* (statt nur des reinen
    Hand Landmarker), weil der zusätzlich zu den Hand-Landmarken auch gleich
    die erkannte Geste (z.B. "Pointing_Up", "Closed_Fist") mitliefert - genau
    das, was ``GestureController`` zum Steuern von Start/Stop braucht. Pro
    Frame bekommt es das aktuelle Kamerabild (Signal ``webcam``), erkennt
    darin die Hand(-Gelenke) + Geste und gibt das Ergebnis als Signal
    ``detector`` an die nachfolgenden Module weiter. Zusätzlich wird eine
    Visualisierung (``galy``) zurückgegeben, damit man die erkannte Hand live
    sieht.
    """

    def __init__(self, outputSignal="detector", model_path="gesture_recognizer.task", num_hands=1):
        """Registriert das Modul beim Framework.

        Parameters
        ----------
        outputSignal : str
            Name des Signals, unter dem das Ergebnis veröffentlicht wird.
        model_path : str
            Pfad zur MediaPipe-Modelldatei ``gesture_recognizer.task``.
        num_hands : int
            Wie viele Hände gleichzeitig gesucht werden (für eine Geste reicht 1).
        """
        super().__init__(
            # Wir abonnieren genau die Signale, die wir lesen:
            #   config = Konfiguration, webcam = aktuelles Kamerabild
            inputSignals=["config", "webcam"],
            # Wir erzeugen genau ein Signal mit dem Namen aus outputSignal.
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="detector",
        )
        self.outputSignal = outputSignal
        self.model_path = model_path
        self.num_hands = num_hands
        self.detector = None  # wird in start() geladen

    def start(self, data):
        """Lädt das Gesture-Recognizer-Modell genau einmal beim Start.

        Hier wird nur vorbereitet, nicht gerechnet: Wir bauen aus der
        Modelldatei einen wiederverwendbaren ``GestureRecognizer`` und merken
        ihn uns in ``self.detector``, damit ihn ``step()`` für jedes Bild
        nutzen kann.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modelldatei '{self.model_path}' nicht gefunden. "
                "Lade sie herunter von: https://storage.googleapis.com/"
                "mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/"
                "gesture_recognizer.task"
            )

        # Modell als Bytes einlesen statt per Pfad zu uebergeben: MediaPipes interner
        # Lader kommt mit Sonderzeichen im Pfad (z.B. das "ue" in "Duesseldorf") nicht
        # klar und wirft sonst einen FileNotFoundError, obwohl die Datei da ist.
        with open(self.model_path, "rb") as f:
            model_bytes = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
        )
        self.detector = vision.GestureRecognizer.create_from_options(options)
        return {}

    def step(self, data):
        """Verarbeitet ein einzelnes Kamerabild.

        Ablauf:
            1. Bild holen und ins richtige Farbformat bringen (BGR -> RGB).
            2. Hände mit MediaPipe erkennen.
            3. Erkannte Hand(-Landmarken) auf einer Ebene visualisieren.
            4. Erkennungsergebnis + Visualisierung zurückgeben.
        """
        frame = data["webcam"]

        # 1. OpenCV liefert das Bild als BGR, MediaPipe erwartet RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 2. Hand- + Gestenerkennung durchführen.
        #    result.hand_landmarks ist eine Liste: pro erkannter Hand 21 Punkte.
        #    result.gestures liefert zusätzlich die erkannte Geste pro Hand.
        result = self.detector.recognize(mp_image)

        # 3. Visualisierung aufbauen.
        height, width = frame.shape[:2]
        galy = GALY()
        # Gleicher Canvas-Name wie im Webcam-Modul ("Main"), sonst entsteht ein
        # zweites, getrenntes Fenster und Hand/Trail landen auf verschiedenen
        # Canvases (Trail waere dann im Hauptfenster unsichtbar).
        galy.canvas("Main", (width, height), (0, 0, 0))  # Zeichenfläche in Bildgröße
        galy.blit("webcam", (0, 0))                       # Kamerabild als Hintergrund
        galy.layer("landmarks")                           # eigene Ebene für die Hand

        # Die Landmarken sind normalisiert (0..1). Diese 2x3-Matrix skaliert sie
        # auf Pixel: x -> x * width, y -> y * height.
        galy.set_layer_affine_mapping(np.array([[width, 0, 0],
                                                [0, height, 0]], dtype=float))

        for hand in result.hand_landmarks:
            draw_hand_landmarks(hand, galy)

        # 4. Ergebnis (für TrailMarker/Preprocessor) + Visualisierung zurückgeben.
        return {self.outputSignal: result, "galy": galy}

    def stop(self, data):
        """Gibt das Modell beim Beenden frei."""
        if self.detector is not None:
            self.detector.close()
