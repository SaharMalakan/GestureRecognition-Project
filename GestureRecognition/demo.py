# Engine = Framework, das die Pipeline Schritt für Schritt (frame-weise) abarbeitet
from SignalHub import Engine, ConfigParser, Webcam
from GestureRecognition.modules import *  # unsere eigenen Module (Handdetector, Trail, ...)
import argparse

def run(parser: argparse.ArgumentParser):
    # zusätzliche Kommandozeilen-Argumente (z.B. python main.py --mode replay)
    parser.add_argument("--mode", action="store", default="none")
    parser.add_argument("--recorder.file", action="store")
    parser.add_argument("--engine.singlestep", action="store_true", default=False)
    parser.add_argument("--webcam.width", required=False)

    # Die Pipeline: Reihenfolge = Reihenfolge der Ausführung pro Frame
    modules = [
        ConfigParser(parser),   # liest config.yml + Kommandozeile
        Webcam(),# liefert das Kamerabild
        HandDetector(),# erkennt Hand + Geste (MediaPipe)
        GestureController(),# steuert Start/Stop der Aufnahme
        TrailMarker(),   # zeichnet die gelbe Spur
        Preprocessor(), # sammelt/normalisiert die Trajektorie
        HMMModule(),  # klassifiziert den Buchstaben
    ]
    engine = Engine(modules=modules, signals={})
    engine.run({})  # startet die Endlosschleife (ein Schritt pro Kamera-Frame)

# nur beim direkten Start (python demo.py), main.py ruft run() selbst auf
if __name__ == "__main__":
    parser = argparse.ArgumentParser("GestureRecognition")
    run(parser)