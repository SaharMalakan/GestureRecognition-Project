import os
import pickle
import time

import cv2
import numpy as np

from SignalHub import GALY, bgr, get_nested_key, Module


class HMMModule(Module):
    """Letztes Modul in der Pipeline - macht aus der Trajektorie eine Geste.

    Kriegt vom Preprocessor GENAU DANN eine fertig normalisierte
    Fingertrajektorie (Signal "preprocessor", sonst None), wenn die Aufnahme
    per Faust gestoppt wurde, und jagt die durch unser trainiertes HMM
    (HMMClassifier). Raus kommt das Label mit dem besten Score. Damit das
    Ergebnis nicht wie frueher jeden Frame neu berechnet wird und flackert,
    wird es fuer ``DISPLAY_SECONDS`` gross/zentriert stehen gelassen.
    """

    DISPLAY_SECONDS = 2.0

    def __init__(self, outputSignal="markov", model_path="data/hmm.pkl", **kwargs):
        """Modul beim Framework anmelden.

        outputSignal: wie das Ergebnis-Signal heißen soll
        model_path: wo das trainierte (gepickelte) HMMClassifier-Modell liegt
        """
        super().__init__(
            # config = Settings, preprocessor = die fertige Trajektorie
            inputSignals=["config", "preprocessor"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="hiddenmarkov",
        )
        self.outputSignal = outputSignal
        self.model_path = model_path
        self.model = None  # kommt erst in start()
        self._model_mtime = None  # mtime von hmm.pkl -> Hot-Reload erkennt neue Modelle
        self.last_result = None  # letztes klassifiziertes {"label", "score"}
        self.last_time = 0.0      # wann das letzte Ergebnis kam (time.time())

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
        if showing_result:
            text = str(self.last_result["label"])
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

        return {self.outputSignal: self.last_result, "galy": galy}

    def stop(self, data):
        """Brauchen wir nicht, das Modell hält keine Ressourcen offen."""
        pass
