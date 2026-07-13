# Gestenerkennung (A–Z in der Luft) — Projektdokumentation

**Team:** Laura Grozdanic, Sahra Ogul, Saharr Malakan  ·  **Modul:** D4.1.1 Machine Perception und Tracking (Prof. Dr. Dennis Müller)  ·  **Datum:** 13.07.2026

> Kurzfassung: Ein HMM-basiertes System erkennt in die Luft „gemalte" Buchstaben
> aus der Fingerspur. Kernbeitrag: durch bessere **Signalverarbeitung**
> (Bogenlängen-Resampling) stieg die Genauigkeit von **~72 % auf ~97 %** —
> **ohne** Trainingsdaten zu löschen.

---

## 1. Ziel & Umfang
Erkennung der 26 Buchstaben A–Z, die mit der Zeigefingerspitze in die Luft
gemalt werden. Umgesetzt sind alle geforderten Bausteine:
Handerkennung + Vorverarbeitung, Labeling-System, Datensatz-Visualisierung,
HMM-Klassifikator und Live-Betrieb.

## 2. Pipeline (Architektur)
```
Webcam ──▶ HandDetector ──▶ TrailMarker (Anzeige der Spur)
                │
                └────────▶ Preprocessor ──▶ HMMModule ──▶ Label + Score
        (MediaPipe          (Trajektorie          (26 HMMs,
         Landmarken)         normalisiert)         argmax Log-Likelihood)
```
Kommunikation zwischen den Modulen über **SignalHub** (jedes Modul abonniert
Signale und veröffentlicht ein Ergebnis-Signal).

## 3. Datensatz
- **26 Klassen** (A–Z), **~30–70 Aufnahmen je Klasse** (insgesamt 1031).
- Aufnahme über Handgesten: Zeigefinger hoch = Start, Faust = Stopp.
- Gespeichert als rohe (T, 2)-Fingerspur pro Aufnahme.
- **Grundsatz:** keine Aufnahmen gelöscht — jede echte Aufnahme bleibt im
  Datensatz (Signaldaten wegwerfen verzerrt; Robustheit kommt aus der
  Verarbeitung, nicht aus dem Aussortieren).

## 4. Vorverarbeitung & Design-Entscheidungen
Jede Trajektorie wird **identisch in Training und Live** verarbeitet:

1. **Bogenlängen-Resampling auf feste 32 Punkte.** Aufnahmen sind
   unterschiedlich lang/schnell — ein langsames „M" hat mehr Punkte als ein
   schnelles. Nach dem Umtasten hat *jede* Geste gleich viele, entlang des
   Pfads gleichmäßig verteilte Punkte → **geschwindigkeits- und
   längeninvariant**. (Grösster Hebel, siehe §6.)
2. **Zentrieren + Skalieren** auf [−1, 1] → unabhängig von Ort und Größe im Bild.

*Verworfene Alternativen (empirisch getestet):* Richtungs-/Winkel-Features und
Glättung — machten es **schlechter** (verstärkten Jitter). Deshalb bewusst
einfache, resampelte (x, y)-Punkte.

## 5. Klassifikator (HMM)
- **Ein GaussianHMM pro Klasse** (hmmlearn). Versteckte Zustände ≈ Phasen der
  Geste (Anfang → Mitte → Ende).
- **`n_states = 12`** (vorher 5). Mehr Zustände wurden erst durch die feste
  Länge (32 Punkte) möglich — vorher deckelte die kürzeste Sequenz die Zahl.
- Klassifikation: eine neue Spur wird unter jedem Modell bewertet
  (Forward-Algorithmus, Log-Likelihood), **argmax** über die Klassen.

## 6. Evaluation & Verlauf
Alle Zahlen auf **identischen Daten, nichts gelöscht**. CV = 5-fache
Kreuzvalidierung (Mittel ± Std), Test = einzelner 80/20-Split.

| Stufe | Verarbeitung | Genauigkeit |
|---|---|---|
| Ausgang | roh, variable Länge, `n_states=5` | 71,2 % (Test) · 71,9 % ± 4,4 % (CV) |
| + Resampling(32) | feste Länge | 81,4 % ± 4,9 % (CV) |
| **+ `n_states=12`** | feste Länge erlaubt mehr Zustände | **94,9 % ± 0,9 % (CV, ~981 Aufn.) · 97,1 % (Test, aktuell 1031 Aufn.)** |

Ablation (verworfen): Richtungs-Features 64 % · Glättung+Richtung 51,8 %.

**Warum Kreuzvalidierung:** Ein einzelner 80/20-Split kann Glück sein — bei uns
zeigte er 91 %, die ehrlichere CV 81 %. Die kleine Streuung (± 0,9 %) belegt,
dass ~95 % **stabil** und kein Zufall sind.

**Confusion Matrix** (`confusion_matrix.png`): Diagonale klar dominant.
Frühere Problem-Buchstaben nach den Änderungen: D 1/6 → 9/10, R 1/6 → 8/8,
A 2/6 → 8/10, M 1/6 → 6/8. Restverwechslungen betreffen formähnliche Paare
(z. B. C↔G), was fachlich plausibel ist.

## 7. Live-Betrieb & Robustheit
- Live läuft die gleiche Verarbeitung wie im Training (garantiert identisches
  Resampling → Training-/Live-Konsistenz per Test bestätigt).
- **Gestensegmentierung (Start/Stopp):** Ein `GestureController` steuert die
  Aufnahme über Handgesten (Zeigefinger hoch = Start, Faust = Stopp). Erst beim
  Stopp wird die **ganze** gemalte Geste **einmal** klassifiziert — also genau
  wie im Training (ganze Gesten, **kein gleitendes Fenster mehr**). Das erkannte
  Label wird groß und zentriert **~2 Sekunden** angezeigt, statt jeden Frame zu
  flackern.
- **Hot-Reload:** Wird während der laufenden Demo neu trainiert (`retrain.py`),
  lädt das HMMModule das frische Modell automatisch nach (mtime-Check) — kein
  Neustart der Demo nötig.
- **Offline vs. Live:** ~97 % gelten für *unsere* Hände; bei einer fremden Hand
  (Prüfer) ist mit weniger zu rechnen → Ausblick: Aufnahmen mehrerer Personen.

## 8. Grenzen & Ausblick
- Aufnahmen mehrerer Personen für bessere Generalisierung.
- **Automatische** bewegungsbasierte Segmentierung (Geste = von Bewegung bis
  Stillstand) — aktuell erfolgt die Segmentierung manuell per Geste
  (Zeigefinger = Start, Faust = Stopp).
- `n_states`/Resample-Länge weiter tunen (leichte Reserve sichtbar).

## 9. Reproduzieren
```cmd
:: Umgebung
call "C:\ProgramData\miniconda3\Scripts\activate.bat" dsai
:: nach neuen Aufnahmen: Datensatz bauen -> trainieren -> auswerten
:: Kurzform (Datensatz bauen + trainieren in EINEM Befehl):
python -m GestureRecognition.retrain
:: oder einzeln:
python -c "from GestureRecognition.labeling import dataset_building; dataset_building('data/dataset.pkl')"
python train_hmm.py
python -m GestureRecognition.visualization evaluate
:: Live
python main.py
```

---
*Bewertungsbezug: §2/§7 → Funktionalität & Robustheit · §3/§4 → Datenqualität ·
§4–§6 → Modellperformance · durchgängig §1–§8 → Verständlichkeit &
Nachvollziehbarkeit.*
