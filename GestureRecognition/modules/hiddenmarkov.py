import os
import pickle
import time

import cv2
import numpy as np

from SignalHub import GALY, bgr, get_nested_key, Module


# Bild-Overlays: wird dieses Label erkannt, kommt statt Text ein Bild.
# label -> Pfad zur PNG-Datei (am besten mit transparentem Hintergrund).
IMAGE_OVERLAYS = {
    "Waluigi": os.path.join(os.path.dirname(__file__), "..", "assets", "waluigi.png"),
}
OVERLAY_MAX_HEIGHT_RATIO = 0.6  # Bild darf max. 60% der Bildhöhe hoch sein


def _overlay_image_alpha(frame, overlay_bgra, x, y):
    """Legt ``overlay_bgra`` (Bild MIT Alphakanal) transparent auf ``frame``.

    x, y = obere linke Ecke, wo das Bild auf dem Frame platziert wird.
    Rechnet pixelweise: ergebnis = vordergrund*alpha + hintergrund*(1-alpha).
    So bleibt der transparente Rand des PNGs auch wirklich durchsichtig,
    statt als schwarzer/weißer Kasten über der Kamera zu liegen.
    """
    h, w = overlay_bgra.shape[:2]

    # Alphakanal (0..255) auf 0.0..1.0 normalisieren, als Spalte für Broadcasting
    alpha = overlay_bgra[:, :, 3:4].astype(float) / 255.0
    foreground = overlay_bgra[:, :, :3].astype(float)

    background = frame[y:y + h, x:x + w].astype(float)
    blended = foreground * alpha + background * (1.0 - alpha)

    frame[y:y + h, x:x + w] = blended.astype(np.uint8)


class HMMModule(Module):
    """Letztes Modul in der Pipeline - macht aus der Trajektorie eine Geste.

    Kriegt vom Preprocessor GENAU DANN eine fertig normalisierte
    Fingertrajektorie (Signal "preprocessor", sonst None), wenn die Aufnahme
    per Faust gestoppt wurde, und jagt die durch unser trainiertes HMM
    (HMMClassifier). Raus kommt das Label mit dem besten Score. Damit das
    Ergebnis nicht wie frueher jeden Frame neu berechnet wird und flackert,
    wird es fuer ``DISPLAY_SECONDS`` gross/zentriert stehen gelassen.

    Für Labels aus ``IMAGE_OVERLAYS`` wird statt des Text-Labels ein Bild
    mittig über das Kamerabild gelegt (z.B. ein Waluigi-Bild bei der
    "Waluigi"-Geste).
    """

    DISPLAY_SECONDS = 2.0

    def __init__(self, outputSignal="markov", model_path="data/hmm.pkl", **kwargs):
        """Modul beim Framework anmelden.

        outputSignal: wie das Ergebnis-Signal heißen soll
        model_path: wo das trainierte (gepickelte) HMMClassifier-Modell liegt
        """
        super().__init__(
            # config = Settings, preprocessor = die fertige Trajektorie,
            # webcam = aktuelles Kamerabild (fürs Bild-Overlay gebraucht)
            inputSignals=["config", "preprocessor", "webcam"],
            outputSchema={
                "type": "object",
                "properties": {outputSignal: {}, "overlayImage": {}},
            },
            name="hiddenmarkov",
        )
        self.outputSignal = outputSignal
        self.model_path = model_path
        self.model = None  # kommt erst in start()
        self._model_mtime = None  # mtime von hmm.pkl -> Hot-Reload erkennt neue Modelle
        self.last_result = None  # letztes klassifiziertes {"label", "score"}
        self.last_time = 0.0      # wann das letzte Ergebnis kam (time.time())
        self._overlay_images = {}  # gecachte, geladene PNGs: label -> BGRA-Array

    def start(self, data):
        """Modell einmal von der Platte laden, bevor's losgeht.

        Klassifiziert wird hier noch nichts, das passiert erst in step().
        Wenn die Datei nicht da ist (z.B. weil noch keiner ein Modell
        trainiert/gespeichert hat), lieber gleich knallen mit einer
        verständlichen Meldung, als dass es später irgendwo komisch crasht.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modelldatei '{self.model_path}' nicht gefunden. "
                "Erst den HMMClassifier trainieren (hmmclassifier.py) und "
                "das trainierte Modell per pickle dort abspeichern."
            )

        self._load_model()

        return {}

    def _load_model(self):
        """Modell (neu) von der Platte laden und den Stand (mtime) merken."""
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        self._model_mtime = os.path.getmtime(self.model_path)

    def _load_overlay_image(self, label):
        """Lädt (einmalig) das PNG-Overlay-Bild für ein Label und merkt es sich.

        Gibt ein BGRA-Array (Bild mit Alphakanal) zurück, oder ``None`` wenn
        es für ``label`` kein Bild gibt oder die Datei fehlt/kaputt ist -
        dann wird später einfach wieder das Text-Label angezeigt.
        """
        if label not in self._overlay_images:
            path = IMAGE_OVERLAYS.get(label)
            image = None
            if path is not None and os.path.exists(path):
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # UNCHANGED = Alphakanal bleibt erhalten
                if image is not None and image.shape[2] != 4:
                    print(f"[hiddenmarkov] '{path}' hat keinen Alphakanal (kein transparentes PNG) - "
                          "zeige stattdessen Text.")
                    image = None
            # auch None merken -> nicht bei jedem Frame erneut von der Platte laden versuchen
            self._overlay_images[label] = image
        return self._overlay_images[label]

    def _reload_if_changed(self):
        """Hot-Reload: läuft die Demo während neu trainiert wird, merken wir
        an der mtime von hmm.pkl, dass es ein frischeres Modell gibt, und
        laden es automatisch nach - kein Neustart der Demo nötig.
        """
        try:
            mtime = os.path.getmtime(self.model_path)
        except OSError:
            return  # Datei kurzzeitig weg (wird gerade neu geschrieben) - nächster Frame
        if mtime == self._model_mtime:
            return
        try:
            self._load_model()
            print(f"[hiddenmarkov] Modell neu geladen: {self.model_path}")
        except Exception:
            pass  # halb geschriebene/kaputte Datei -> altes Modell behalten, naechster Frame erneut

    def step(self, data):
        """Pro Frame: bei fertiger Trajektorie klassifizieren, Ergebnis anzeigen.

        ``trajectory`` ist nur in dem einen Frame nach dem Faust-Stop gesetzt
        (siehe Preprocessor) - wir klassifizieren also GENAU EINMAL pro
        Aufnahme, statt wie frueher jeden Frame neu (das flackerte, weil sich
        die laufend gesammelte Trajektorie staendig aenderte).
        """
        self._reload_if_changed()
        trajectory = data.get("preprocessor")

        if trajectory is not None:
            # decision_function/predict wollen eigentlich eine ganze Liste von
            # Sequenzen (typisch sklearn-mäßig), wir haben aber nur eine einzige
            # -> deswegen die [trajectory] mit den eckigen Klammern
            scores = self.model.decision_function([trajectory])[0]
            label = self.model.predict([trajectory])[0]
            score = float(np.max(scores))  # höchster Score = wie sicher sich das Modell ist
            self.last_result = {"label": label, "score": score}
            self.last_time = time.time()

        # eigene Ebene fürs Text-Zeichnen, NICHT die "landmarks"-Ebene vom
        # Handdetector mitbenutzen - die hat ein Affine-Mapping für 0..1
        # Koordinaten drauf, wir wollen aber ganz normale Pixel-Koordinaten
        galy = GALY()
        galy.layer("hiddenmarkov")

        config = data.get("config", {})
        width = get_nested_key("webcam.width", config, default=640)
        height = get_nested_key("webcam.height", config, default=360)

        showing_result = (
            self.last_result is not None
            and (time.time() - self.last_time) < self.DISPLAY_SECONDS
        )

        overlay_image = None  # gefüllt, wenn wir statt Text ein Bild zeigen
        if showing_result:
            label = str(self.last_result["label"])
            frame = data.get("webcam")
            template = self._load_overlay_image(label) if frame is not None else None

            if template is not None:
                # Bild proportional verkleinern, damit es nicht über den
                # Bildschirm hinausragt (max. OVERLAY_MAX_HEIGHT_RATIO der Höhe).
                frame_h, frame_w = frame.shape[:2]
                scale = min(
                    (frame_h * OVERLAY_MAX_HEIGHT_RATIO) / template.shape[0],
                    frame_w / template.shape[1],
                )
                new_w = max(1, int(template.shape[1] * scale))
                new_h = max(1, int(template.shape[0] * scale))
                resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # mittig platzieren, genau wie vorher der Text
                x = max(0, (frame_w - new_w) // 2)
                y = max(0, (frame_h - new_h) // 2)

                overlay_image = frame.copy()
                _overlay_image_alpha(overlay_image, resized, x, y)
                galy.blit("overlayImage", (0, 0))
            else:
                text = label
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)
                org = (max(0, width // 2 - text_w // 2), height // 2 + text_h // 2)
                galy.putText(text, org, fontScale=3.0, color=bgr("#00FF00"), thickness=6)
        else:
            galy.putText(
                "Zeigefinger hoch = Buchstaben malen",
                (10, height - 15),  # unten links, ähnlich wie beim Labeling-Tool
                fontScale=0.7,
                color=bgr("#00FF00"),
                thickness=2,
            )

        return {self.outputSignal: self.last_result, "galy": galy, "overlayImage": overlay_image}

    def stop(self, data):
        """Brauchen wir nicht, das Modell hält keine Ressourcen offen."""
        pass
