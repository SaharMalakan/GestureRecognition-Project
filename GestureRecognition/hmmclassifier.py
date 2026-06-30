import pickle
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMClassifier:
    """
    HMM-basierter Klassifikator fuer 2D-Gestentrajektorien.

    Grundidee (Vorlesung):
    ----------------------
    - Beobachtung Y = die 2D-Trajektorienpunkte (x, y) der Fingerspitze.
    - Versteckte Zustaende = Phasen der Geste (z. B. Anfang / Mitte / Ende).
    - Pro Klasse wird EIN eigenes HMM trainiert (hmmlearn.GaussianHMM).
    - ``fit`` trainiert jedes HMM: hmmlearn schaetzt aus den Beispiel-Sequenzen
      die Uebergangswahrscheinlichkeit p(x_t | x_{t-1}) (Zustand -> Zustand) und
      die Emissionswahrscheinlichkeit p(y_t | x_t) (Zustand -> Beobachtung).
    - Eine neue Sequenz wird unter JEDEM Klassenmodell bewertet: der Score ist
      die Log-Likelihood log P(Y | Modell), berechnet vom Forward-Algorithmus
      (hmmlearn ``model.score``).
    - Klassifikation = argmax ueber die Klassen dieser Log-Likelihoods.

    Datenformat:
    ------------
    - Datensatz: dict ``{ label: [seq0, seq1, ...] }``, jede Sequenz ein
      normalisiertes (T, 2)-numpy-Array (siehe ``labeling.normalize_trajectory``).
    - ``predict`` / ``decision_function`` bekommen eine LISTE von (T, 2)-Arrays.
      Eine einzelne Live-Trajektorie wird vom Aufrufer als ``[traj]`` uebergeben.

    Speicherung:
    ------------
    - ``save``/``load`` mit pickle speichern bzw. laden das ganze Objekt
      (Klassen + Modelle + Hyperparameter) in einer ``.pkl``-Datei.
    """

    def __init__(self, n_states=5, n_iter=20, covariance_type="diag", random_state=42):
        # Hyperparameter merken. n_states = Anzahl versteckter Zustaende (Phasen).
        # n_states koennte optional per Grid Search getunt werden -> bewusst NICHT
        # im Standard-Pfad, damit der Code einfach bleibt.
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type   # "diag" = robust, je Zustand 2 Varianzen
        self.random_state = random_state         # feste Seed -> reproduzierbar
        self.models = {}                         # label -> trainiertes HMM (oder None)
        self.labels = []                         # feste, sortierte Klassenreihenfolge

    def fit(self, dataset):
        """Trainiert pro Klasse ein GaussianHMM. dataset = {label: [(T,2)-Arrays]}."""
        # Klassen in sortierter Reihenfolge -> konsistent mit evaluate_classifier.
        for label in sorted(dataset.keys()):
            # Jede Sequenz sauber als float-Array (T,2) -> hmmlearn rechnet mit float.
            seqs = [np.asarray(s, dtype=float) for s in dataset[label]]

            # Alle Sequenzen einer Klasse untereinander stapeln: X hat (sum_T, 2).
            X = np.vstack(seqs)
            # lengths in DERSELBEN Reihenfolge -> sum(lengths) == len(X). So weiss
            # hmmlearn, wo eine Sequenz endet (keine falschen Uebergaenge dazwischen).
            lengths = [len(s) for s in seqs]

            # n_states darf nie laenger als die kuerzeste Sequenz dieser Klasse sein.
            n = min(self.n_states, min(lengths))

            model = GaussianHMM(
                n_components=n,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            try:
                # ConvergenceWarning u. a. von hmmlearn unterdruecken -> saubere Konsole.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # hmmlearn schaetzt die Modell-Parameter (Uebergaenge + Emissionen) aus den Daten.
                    model.fit(X, lengths)
                self.models[label] = model
            except Exception:
                # Training fehlgeschlagen (z. B. singulaere Kovarianz) -> None.
                # Diese Klasse bekommt spaeter Score -inf und wird nie gewaehlt.
                print(f"Warnung: Modell fuer Klasse '{label}' konnte nicht trainiert werden.")
                self.models[label] = None

        # Feste Klassenreihenfolge fuer decision_function (Spalten) und predict.
        self.labels = sorted(self.models)
        return self

    def decision_function(self, sequences):
        """Log-Likelihood je Sequenz und Klasse -> Array (n_sequences, n_classes)."""
        # Start mit -inf: fehlgeschlagene/ungueltige Modelle bleiben automatisch -inf.
        scores = np.full((len(sequences), len(self.labels)), -np.inf)
        for i, seq in enumerate(sequences):
            seq = np.asarray(seq, dtype=float)
            for j, label in enumerate(self.labels):
                model = self.models[label]
                if model is None:
                    continue  # nicht trainiert -> bleibt -inf
                try:
                    # Forward-Algorithmus: s = log P(Y | Modell).
                    s = model.score(seq)
                    if np.isfinite(s):       # NaN / -inf nicht uebernehmen
                        scores[i, j] = s
                except Exception:
                    pass  # Score nicht berechenbar -> bleibt -inf
        return scores

    def predict(self, sequences):
        """Bestes Label je Sequenz. sequences = LISTE von (T,2)-Arrays."""
        scores = self.decision_function(sequences)
        out = []
        for row in scores:
            # Fallback: sind alle Scores -inf, ist keine Klasse sinnvoll -> festes
            # Label, statt dass argmax stillschweigend Index 0 waehlt.
            if not np.any(np.isfinite(row)):
                out.append(self.labels[0])
            else:
                # Klassifikation = argmax der Log-Likelihoods.
                out.append(self.labels[int(np.argmax(row))])
        return out

    def save(self, path):
        """Speichert das ganze Objekt (Klassen + Modelle + Hyperparameter)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Laedt eine fertige HMMClassifier-Instanz (z. B. fuer den Live-Modus)."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def train_test_split(dataset, test_ratio=0.2, seed=42):
        """Teilt pro Klasse in Trainings- und Testsequenzen (Standard 80/20)."""
        rng = np.random.default_rng(seed)        # feste Seed -> reproduzierbarer Split
        train, test = {}, {}
        for label, seqs in dataset.items():
            idx = rng.permutation(len(seqs))     # Reihenfolge je Klasse mischen
            n_test = int(len(seqs) * test_ratio)
            n_train = max(1, len(seqs) - n_test)  # mind. 1 Sequenz zum Trainieren
            train[label] = [seqs[k] for k in idx[:n_train]]
            test[label] = [seqs[k] for k in idx[n_train:]]
        return train, test
