from SignalHub import GALY, Module, get_nested_key
from collections import deque
import numpy as np

class TrailMarker(Module):
    """
    Modul zum Zeichnen einer Spur anhand der Bewegung eines Fingers.

    Die Position eines bestimmten Finger-Landmarks wird über mehrere Frames
    hinweg gespeichert. Aus diesen Punkten kann anschließend eine Linie
    erzeugt werden, die den Bewegungsverlauf des Fingers visualisiert.

    Ziel ist es, die Verarbeitung der Landmark-Daten sowie die Verwaltung
    eines Zustands über mehrere Frames hinweg selbst zu implementieren.
    """

    def __init__(self, outputSignal="trailmarker"):
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
        Da dieses Modul keine eigenen Daten erzeugt, reicht beispielsweise:

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
            name="trailmarker",
        )

    def start(self, data):
        """
        Initialisierung des Modulzustands.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, alle Variablen vorzubereiten, die während der
        Laufzeit des Moduls benötigt werden.

        Hinweise
        --------
        - Lese benötigte Parameter aus der Konfiguration.
        - Bestimme beispielsweise, welcher Finger verfolgt werden soll.
        - Lege eine Datenstruktur an, in der mehrere vergangene
          Fingerpositionen gespeichert werden können,
          z.B. :class:`collections.deque` mit einer maximalen Größe.
        - Diese Historie wird später verwendet, um eine Spur zu zeichnen.
        - Speichere aus der Konfiguration weitere benötigte Parameter,
          z.B. Finger-Index, maximale Anzahl verlorener Frames oder
          Webcam-Parameter.
        - Für den Zugriff auf verschachtelte Konfigurationswerte kann
          :meth:`get_nested_key` verwendet werden.

        .. tip::
           Eine ``deque`` ist ideal für Trajektorien,
           da sie effizient alte Punkte entfernt.

        .. note::
           Initialisiere hier nur Zustände und Parameter,
           keine eigentliche Verarbeitung.

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
        self.finger_idx = get_nested_key("trailmarker.finger_idx", config, default=8)   # 8 = Zeigefinger-Spitze
        self.max_lost = get_nested_key("trailmarker.max_lost", config, default=10)      # Toleranz bei Hand-Verlust
        # Gross genug für einen ganzen "gemalten" Buchstaben (Trail wird bei
        # jedem neuen Start über die Geste sowieso geleert, siehe step()).
        trail_length = get_nested_key("trailmarker.trail_length", config, default=300)
        self.trail = deque(maxlen=trail_length)   # speichert die letzten Fingerpositionen
        self.lost_frames = 0   # Zähler: wie viele Frames am Stück keine Hand erkannt
        # Für die Skalierung der normalisierten (0..1) Finger-Koordinaten auf
        # Pixel - genau wie im HandDetector (siehe eigene Layer-Mapping unten).
        self.width = get_nested_key("webcam.width", config, default=1280)
        self.height = get_nested_key("webcam.height", config, default=720)
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, die aktuelle Position eines Fingers zu bestimmen,
        diese Position in einer Trajektorie zu speichern und daraus
        eine visuelle Spur zu erzeugen.

        Hinweise
        --------
        - Greife auf das ``detector`` Signal zu, um erkannte Hände und
          deren Landmarken zu erhalten.
        - Falls keine Hand erkannt wurde, kann beispielsweise ein Zähler
          für verlorene Frames erhöht werden.
        - Wird eine Hand erkannt, kann die Landmarke des gewünschten
          Fingers extrahiert werden.
        - Die Position kann zur bestehenden Trajektorie hinzugefügt werden.
        - Zwischen aufeinanderfolgenden Punkten können Linien gezeichnet
          werden, um eine Spur darzustellen.
        - Für die Visualisierung kann :meth:`line` der :class:`GALY`
          verwendet werden.

        .. tip::
          Typischer Ablauf:
           1. Landmark extrahieren
           2. Punkt speichern
           3. Trajektorie aktualisieren
           4. Linien zwischen Punkten zeichnen

        .. warning::
            Achte darauf, dass:
              - keine leeren Landmark-Daten verarbeitet werden
              - die Trajektorie nicht unendlich wächst
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
            Um die Zeichenoperationen auszuführen, sollte ein
            :class:`GALY` Objekt zurückgegeben werden.

            Beispiel:

            ``return { ..., "galy": galy}``
        """
        galy = GALY()
        # Eigene Ebene mit eigenem Pixel-Mapping - unabhängig davon, ob/welche
        # andere Module ihre Ebene gerade gesetzt haben. Ohne das würden die
        # normalisierten (0..1) Koordinaten ungeskaliert (~1 Pixel) gezeichnet
        # und die Spur wäre praktisch unsichtbar. "alwaysVisible" heisst
        # ausserdem, dass die Spur nicht über das Layer-Menu ausgeblendet
        # werden kann.
        galy.layer("trailmarker", alwaysVisible=True)
        galy.set_layer_affine_mapping(np.array([[self.width, 0, 0],
                                                 [0, self.height, 0]], dtype=float))

        gc = data.get("gesturecontroller", {})   # Zustand von GestureController

        if gc.get("reset"):
            self.trail.clear()   # neue Aufnahme -> alte Spur weg

        # Nur während der Aufnahme (Zeigefinger hoch, siehe GestureController)
        # überhaupt Punkte sammeln/zeichnen - genau wie beim Labeling-Tool.
        if not gc.get("recording"):
            return {"galy": galy}

        detector = data.get("detector")

        if not detector or not detector.hand_landmarks:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost:
                self.trail.clear()   # Hand zu lange weg -> Spur verwerfen
            return {"galy": galy}

        self.lost_frames = 0
        lm = detector.hand_landmarks[0][self.finger_idx]   # Position der Fingerspitze
        self.trail.append((lm.x, lm.y))   # Punkt zur Spur hinzufügen

        trail_list = list(self.trail)
        for i in range(1, len(trail_list)):
            galy.line(trail_list[i - 1], trail_list[i], (0, 255, 255), 2)   # gelbe Linie
        if trail_list:
            galy.circle(trail_list[-1], 6, (0, 255, 255), -1)   # Punkt an Fingerspitze

        return {"galy": galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf Ressourcen freizugeben oder interne
        Zustände zurückzusetzen.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber sinnvoll sein,
           wenn Zustände explizit zurückgesetzt werden sollen.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass