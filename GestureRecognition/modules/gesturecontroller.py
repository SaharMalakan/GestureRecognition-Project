from SignalHub import GALY, bgr, get_nested_key, Module


class GestureController(Module):
    """Steuert Start/Stop der Aufnahme über Handgesten - genau wie beim Labeling.

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

    START_GESTURE = "Pointing_Up"
    STOP_GESTURE = "Closed_Fist"

    def __init__(self, outputSignal="gesturecontroller"):
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="gesturecontroller",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        config = data.get("config", {})
        self.trigger_frames = get_nested_key("gesturecontroller.trigger_frames", config, default=3)
        self.state = "idle"
        self.last_gesture = None
        self.streak = 0
        return {}

    def step(self, data):
        detector = data.get("detector")
        gesture = "None"
        if detector is not None and getattr(detector, "gestures", None):
            gesture = detector.gestures[0][0].category_name

        # Geste entprellen: erst zaehlen, wenn sie ein paar Frames stabil ist.
        self.streak = self.streak + 1 if gesture == self.last_gesture else 1
        self.last_gesture = gesture
        stable = self.streak >= self.trigger_frames

        reset, finished = False, False
        if self.state == "idle" and gesture == self.START_GESTURE and stable:
            self.state = "recording"
            reset = True
        elif self.state == "recording" and gesture == self.STOP_GESTURE and stable:
            self.state = "idle"
            finished = True

        result = {
            "state": self.state,
            "gesture": gesture,
            "recording": self.state == "recording",
            "reset": reset,
            "finished": finished,
        }

        galy = GALY()
        galy.layer("gesturecontroller")
        hinweis = "Zeigefinger hoch = Start" if self.state == "idle" else "Faust = Stop"
        galy.putText(
            f"[{self.state}] {hinweis}   (Geste: {gesture})",
            (10, 30),
            fontScale=0.7,
            color=bgr("#C8C8C8") if self.state == "idle" else bgr("#FF3232"),
            thickness=2,
        )

        return {self.outputSignal: result, "galy": galy}

    def stop(self, data):
        pass
