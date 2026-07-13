from SignalHub import GALY, bgr, get_nested_key, Module


class GestureController(Module):
    """Steuert Start/Stop der Aufnahme über Handgesten - genau wie beim Labeling

    Zeigefinger hoch (``Pointing_Up``) -> Aufnahme startet.
    Faust (``Closed_Fist``)            -> Aufnahme stoppt.

    Eine Geste muss ein paar Frames stabil sein (Entprellung gegen Flackern),
    bevor sie zählt. Das Ergebnis ist ein einzelnes Signal (Dict) mit dem
    aktuellen Zustand, das TrailMarker/Preprocessor/HMMModule auswerten:

    - ``recording``: True, solange gerade aufgenommen wird.
    - ``reset``:     True genau in dem Frame, in dem eine neue Aufnahme startet
                     (Signal zum Leeren der alten Trajektorie).
    - ``finished``:  True genau in dem Frame, in dem die Aufnahme stoppt
                     (Signal zum Klassifizieren der fertigen Trajektorie).
    """

    START_GESTURE = "Pointing_Up" # Name kommt so vom MediaPipe Gesture Recognizer
    STOP_GESTURE = "Closed_Fist"

    def __init__(self, outputSignal="gesturecontroller"):
        super().__init__(
            inputSignals=["config", "detector"],   # config = Settings, detector = erkannte Hand+Geste
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="gesturecontroller",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        config = data.get("config", {})
        # so viele Frames muss die Geste halten, bevor sie zählt (gegen Flackern)
        self.trigger_frames = get_nested_key("gesturecontroller.trigger_frames", config, default=3)
        self.state = "idle"  # idle = wartet auf Start-Geste
        self.last_gesture = None  # letzte gesehene Geste (für Entprellung)
        self.streak = 0  # wie viele Frames in Folge gleiche Geste
        return {}

    def step(self, data):
        detector = data.get("detector")
        gesture = "None"   # Standard: keine Geste erkannt
        if detector is not None and getattr(detector, "gestures", None):
            gesture = detector.gestures[0][0].category_name   # Name der erkannten Geste

        # Geste entprellen: erst zählen, wenn sie ein paar Frames stabil ist.
        self.streak = self.streak + 1 if gesture == self.last_gesture else 1
        self.last_gesture = gesture
        stable = self.streak >= self.trigger_frames   # Geste lange genug gehalten?

        # --- Zustandsautomat: idle <-> recording ---
        reset, finished = False, False
        if self.state == "idle" and gesture == self.START_GESTURE and stable:
            self.state = "recording"
            reset = True       # Signal an andere Module: Trajektorie leeren
        elif self.state == "recording" and gesture == self.STOP_GESTURE and stable:
            self.state = "idle"
            finished = True     # Signal an andere Module: jetzt klassifizieren

        # Ergebnis, das andere Module (TrailMarker, Preprocessor, HMMModule) lesen
        result = {
            "state": self.state,
            "gesture": gesture,
            "recording": self.state == "recording",
            "reset": reset,
            "finished": finished,
        }

        # Text-Anzeige im Kamerabild (aktueller Zustand + Geste)
        galy = GALY()
        galy.layer("gesturecontroller")
        hinweis = "Zeigefinger hoch = Start" if self.state == "idle" else "Faust = Stop"
        galy.putText(
            f"[{self.state}] {hinweis}   (Geste: {gesture})",
            (10, 30),
            fontScale=0.7,
            color=bgr("#C8C8C8") if self.state == "idle" else bgr("#FF3232"),  # grau/rot
            thickness=2,
        )

        return {self.outputSignal: result, "galy": galy}

    def stop(self, data):
        pass   # nichts aufzuräumen
