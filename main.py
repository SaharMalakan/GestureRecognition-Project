import argparse  # zum Einlesen von Kommandozeilen-Argumenten (z.B. --mode)

from GestureRecognition import run  # startet die ganze Pipeline (siehe demo.py)

parser = argparse.ArgumentParser("GestureRecognition")  # Argument-Parser vorbereiten

def main():
    run(parser)  # Einstiegspunkt: startet Webcam + Pipeline

#zum ausführen 
if __name__ == "__main__":
    main()
