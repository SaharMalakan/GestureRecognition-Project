import os
import pickle

import numpy as np

from SignalHub import GALY, bgr, get_nested_key, Module


class HMMModule(Module):
    """Letztes Modul in der Pipeline - macht aus der Trajektorie eine Geste.

    Kriegt vom Preprocessor pro Frame die fertig normalisierte
    Fingertrajektorie rein (Signal "preprocessor") und jagt die durch
    unser trainiertes HMM (HMMClassifier). Raus kommt das Label mit dem
    besten Score, plus ein bisschen Text im Bild zur Kontrolle.
    """

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

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        return {}

    def step(self, data):
        """Pro Frame: Trajektorie nehmen, Modell fragen, Ergebnis anzeigen."""
        trajectory = data.get("preprocessor")

        # solange der Preprocessor noch nicht genug Punkte hat, kommt hier
        # None an - dann gibt's halt noch nichts zu erkennen, einfach warten
        if trajectory is None:
            return {self.outputSignal: None}

        # decision_function/predict wollen eigentlich eine ganze Liste von
        # Sequenzen (typisch sklearn-mäßig), wir haben aber nur eine einzige
        # -> deswegen die [trajectory] mit den eckigen Klammern
        scores = self.model.decision_function([trajectory])[0]
        label = self.model.predict([trajectory])[0]
        score = float(np.max(scores))  # höchster Score = wie sicher sich das Modell ist

        result = {"label": label, "score": score}

        # eigene Ebene fürs Text-Zeichnen, NICHT die "landmarks"-Ebene vom
        # Handdetector mitbenutzen - die hat ein Affine-Mapping für 0..1
        # Koordinaten drauf, wir wollen aber ganz normale Pixel-Koordinaten
        galy = GALY()
        galy.layer("hiddenmarkov")

        config = data.get("config", {})
        height = get_nested_key("webcam.height", config, default=360)

        galy.putText(
            f"Geste: {label} ({score:.1f})",
            (10, height - 15),  # unten links, ähnlich wie beim Labeling-Tool
            fontScale=0.7,
            color=bgr("#00FF00"),
            thickness=2,
        )

        return {self.outputSignal: result, "galy": galy}

    def stop(self, data):
        """Brauchen wir nicht, das Modell hält keine Ressourcen offen."""
        pass
