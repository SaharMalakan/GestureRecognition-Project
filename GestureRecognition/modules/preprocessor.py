from SignalHub import get_nested_key, Module
import numpy as np

from GestureRecognition.labeling import resample_trajectory as _resample_trajectory
from GestureRecognition.labeling import _center_and_scale


def resample_trajectory(points, n=32):
    """Trajektorie per Bogenlänge auf ``n`` feste Punkte umtasten.

    Muss IDENTISCH zu labeling.resample_trajectory sein - sonst passt die
    Live-Eingabe nicht zu den Trainingsdaten. Macht die Geste geschwindigkeits-
    und längeninvariant (gleich viele, gleichmäßig verteilte Punkte pro Geste).
    """
    return _resample_trajectory(points, n)


class Preprocessor(Module):
    """
    Modul zur Vorverarbeitung von Fingertrajektorien.

    Dieses Modul verarbeitet die vom Handdetektor gelieferten Landmarken
    und extrahiert daraus die Bewegung eines bestimmten Fingers über
    mehrere Frames hinweg.

    Ziel ist es, eine Trajektorie zu sammeln, diese zu normalisieren
    und anschließend als Eingabe für nachfolgende Module bereitzustellen.
    """

    def __init__(self, outputSignal="preprocessor"):
        """
        Konstruktor des Moduls.

        Ziel ist es, das Modul beim Framework korrekt zu registrieren.

        Hinweise
        --------
        - Ein Modul muss definieren, **welche Signale es empfangen möchte**.
        - Diese werden über ``inputSignals`` angegeben.
        - Nur Signale, die hier subscribed werden, erscheinen später im
          ``data`` Dictionary der Methoden :meth:`start` und :meth:`step`.

        Für dieses Modul werden unter anderem folgende Signale benötigt:

        - ``config`` : Systemkonfiguration
        - ``detector`` : Ergebnisse der Handdetektion

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``preprocessor``.

        Dieses Signal enthält entweder eine normalisierte Trajektorie
        oder ``None``, falls noch nicht genügend Daten gesammelt wurden.

        Beispiel:

        ``outputSchema={"type": "object", "properties": {outputSignal: {}}}``

        .. note::
           Die Basisklasse :class:`Module` erwartet beim Aufruf von
           ``super().__init__`` unter anderem:

           - ``inputSignals``
           - ``outputSchema``
           - ``name`` des Moduls

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.
        """
        super().__init__(
            inputSignals=["config", "detector", "gesturecontroller"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        """
        Initialisierung des Modulzustands.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, alle benötigten Parameter aus der Konfiguration zu
        lesen und interne Datenstrukturen vorzubereiten.

        Hinweise
        --------
        - Lese relevante Parameter aus der Konfiguration, z.B.
          den zu verfolgenden Finger.
        - Lege eine Datenstruktur an, um mehrere vergangene
          Fingerpositionen zu speichern, z.B. :class:`collections.deque`
          mit einer maximalen Größe.
        - Speichere außerdem Parameter wie die maximale Anzahl
          verlorener Frames oder die minimale Anzahl benötigter Punkte.
        - Zum Zugriff auf verschachtelte Konfigurationswerte kann
          :meth:`get_nested_key` verwendet werden.

        .. tip::
            Eine ``deque`` mit fester Länge ist ideal für Trajektorien,
            da alte Punkte automatisch verworfen werden.

        .. note::
            Trenne klar zwischen:
              - Initialisierung von Parametern (``start``)
              - Verarbeitung von Daten (``step``)

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks. Enthält unter anderem das
            Signal ``config``.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """
        config = data.get("config", {})
        self.finger_idx = get_nested_key("preprocessor.finger_idx", config, default=8)   # 8 = Zeigefinger-Spitze
        self.max_lost = get_nested_key("preprocessor.max_lost", config, default=10)
        # Gleicher Schwellwert wie labeling.MIN_SEQUENCE_LEN - kürzere
        # Aufnahmen würden auch beim Trainingsdatensatz verworfen.
        self.min_points = get_nested_key("preprocessor.min_points", config, default=8)
        self.n_resample = get_nested_key("preprocessor.n_resample", config, default=32)   # Ziel-Punktzahl
        # Keine feste Länge/Deque mehr: die Trajektorie wird bei Start (über
        # GestureController) geleert und wächst bis zum Stop - genau wie beim
        # Labeling, wo auch die GANZE Aufnahme zwischen Start und Stop zählt.
        self.trajectory = []   # gesammelte (x, y) Punkte der aktuellen Aufnahme
        self.lost_frames = 0   # Zähler: Frames ohne erkannte Hand
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, eine Fingerposition aus den erkannten Landmarken
        zu extrahieren und diese in einer Trajektorie zu speichern.

        Hinweise
        --------
        - Greife auf das ``detector`` Signal zu, um erkannte
          Handlandmarks zu erhalten.
        - Falls keine Hand erkannt wurde, sollte ein interner
          Zähler für verlorene Frames erhöht werden.
        - Wird eine Hand erkannt, kann die Landmarke des gewünschten
          Fingers extrahiert werden.
        - Die Position dieses Fingers kann anschließend in einer
          Trajektorie gespeichert werden.
        - Sobald genügend Punkte gesammelt wurden, kann die
          Trajektorie weiterverarbeitet werden.

        Mögliche Verarbeitungsschritte:

        - Umwandlung der gespeicherten Punkte in ein
          :class:`numpy.ndarray`
        - Berechnung eines Zentrums der Trajektorie
        - Skalierung oder Normalisierung der Punkte

        .. tip::
            Arbeite schrittweise:
              1. Prüfen, ob Landmarken vorhanden sind
              2. Fingerposition extrahieren
              3. In Trajektorie speichern
              4. Optional normalisieren

        .. warning::
            Achte darauf, dass:
              - genügend Punkte vorhanden sind
              - keine fehlerhaften Frames verarbeitet werden
              - verlorene Frames sinnvoll behandelt werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Gibt entweder ``None`` oder eine normalisierte Trajektorie
            zurück.

            Beispiel:

            ``return {outputSignal: trajectory}``
        """
        gc = data.get("gesturecontroller", {})   # Zustand von GestureController

        if gc.get("reset"):
            self.trajectory = []   # neue Aufnahme -> alte Punkte weg
            self.lost_frames = 0

        # Nur während der Aufnahme (Zeigefinger hoch) Punkte sammeln - der
        # Faust-Stop friert die Trajektorie ein, damit sie unten normalisiert
        # werden kann, ohne dass die letzten (Faust-)Frames sie verfälschen.
        if gc.get("recording"):
            detector = data.get("detector")
            if not detector or not detector.hand_landmarks:
                self.lost_frames += 1
                if self.lost_frames > self.max_lost:
                    self.trajectory = []   # Hand zu lange weg -> verwerfen
            else:
                self.lost_frames = 0
                lm = detector.hand_landmarks[0][self.finger_idx]   # Fingerspitze
                self.trajectory.append((lm.x, lm.y))

        # Erst wenn die Aufnahme gerade gestoppt wurde (Faust erkannt), wird
        # EINMAL normalisiert und weitergegeben - identisch zu
        # labeling.normalize_trajectory (Training == Live!).
        if gc.get("finished") and len(self.trajectory) >= self.min_points:
            points = np.array(self.trajectory)
            points = resample_trajectory(points, self.n_resample)   # feste Punktzahl
            points = _center_and_scale(points)                       # zentrieren + skalieren
            return {self.outputSignal: points}

        return {self.outputSignal: None}   # noch nicht fertig -> nichts zu klassifizieren

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf interne Zustände zurückzusetzen
        oder Ressourcen freizugeben.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber relevant werden,
           wenn interne Zustände explizit zurückgesetzt werden sollen.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass