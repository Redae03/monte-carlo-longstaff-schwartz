import time
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QComboBox, QPushButton, QGroupBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QFrame
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from MonteCarloApp.core.black_scholes import bs_price


# ---------- Helpers ----------
def _card(title: str) -> QFrame:
    c = QFrame()
    c.setObjectName("MCCard")
    lay = QVBoxLayout(c)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(6)

    t = QLabel(title)
    t.setObjectName("MCCardTitle")
    v = QLabel("-")
    v.setObjectName("MCCardValue")

    lay.addWidget(t)
    lay.addWidget(v)
    c._value_label = v  # attach
    return c


def _mc_stats(samples: np.ndarray):
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    mean = float(np.mean(samples))
    var = float(np.var(samples, ddof=1)) if n > 1 else 0.0
    std = float(np.sqrt(var))
    se = std / np.sqrt(n) if n > 0 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return mean, var, (ci_low, ci_high)


def _simulate_ST(S0, T, r, sigma, Z):
    S0 = float(S0); T = float(T); r = float(r); sigma = float(sigma)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)



def _payoff(ST, K, opt_type):
    if opt_type == "call":
        return np.maximum(ST - float(K), 0.0)
    return np.maximum(float(K) - ST, 0.0)


# ---------- Main Tab ----------
class MethodsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        # ----- Params box (comme l'image) -----
        box = QGroupBox("Paramètres (option européenne)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        self.in_S0 = QLineEdit("100")
        self.in_K = QLineEdit("105")
        self.in_T = QLineEdit("1")
        self.in_r = QLineEdit("0.05")
        self.in_sigma = QLineEdit("0.2")
        self.in_N = QLineEdit("50000")

        self.cmb_type = QComboBox()
        self.cmb_type.addItems(["call", "put"])

        row = 0
        grid.addWidget(QLabel("Prix initial (S0) :"), row, 0); grid.addWidget(self.in_S0, row, 1)
        grid.addWidget(QLabel("Strike (K) :"),        row, 2); grid.addWidget(self.in_K,  row, 3)
        row += 1
        grid.addWidget(QLabel("Volatilité (σ) :"),    row, 0); grid.addWidget(self.in_sigma, row, 1)
        grid.addWidget(QLabel("Taux sans risque (r) :"),row,2); grid.addWidget(self.in_r,    row, 3)
        row += 1
        grid.addWidget(QLabel("Échéance (T) :"),      row, 0); grid.addWidget(self.in_T, row, 1)
        grid.addWidget(QLabel("Simulations (N) :"),   row, 2); grid.addWidget(self.in_N, row, 3)
        row += 1
        grid.addWidget(QLabel("Type :"),              row, 0); grid.addWidget(self.cmb_type, row, 1)

        self.btn = QPushButton("Comparer")
        self.btn.clicked.connect(self.on_run)
        grid.addWidget(self.btn, row, 3)

        root.addWidget(box)

        # ----- Top cards (Standard / Améliorée) -----
        top = QHBoxLayout()
        top.setSpacing(12)
        self.card_std = _card("Méthode Standard")
        self.card_imp = _card("Méthode Améliorée (Contrôle Variance)")

        top.addWidget(self.card_std, 1)
        top.addWidget(self.card_imp, 1)
        root.addLayout(top)

        # ----- Middle: table + plot -----
        mid = QHBoxLayout()
        mid.setSpacing(12)

        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Méthode", "Erreur", "Temps (s)", "Variance"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        from PyQt5.QtWidgets import QHeaderView
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Plot
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Theme (light comme l'image 1)
        bg = "#f2f2f2"
        gridc = "#d9d9d9"
        txt = "#000000"
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor("white")
        for spine in self.ax.spines.values():
            spine.set_color("#000000")
        self.ax.tick_params(colors=txt)
        self.ax.title.set_color(txt)
        self.ax.xaxis.label.set_color(txt)
        self.ax.yaxis.label.set_color(txt)
        self.ax.grid(True, color=gridc, alpha=0.6)

        # Wrap cards like panels
        table_wrap = QFrame()
        table_wrap.setObjectName("MCPanel")
        tlay = QVBoxLayout(table_wrap)
        tlay.setContentsMargins(10, 10, 10, 10)
        tlay.addWidget(self.table)

        plot_wrap = QFrame()
        plot_wrap.setObjectName("MCPanel")
        play = QVBoxLayout(plot_wrap)
        play.setContentsMargins(10, 10, 10, 10)
        play.addWidget(self.canvas)

        mid.addWidget(table_wrap, 2)
        mid.addWidget(plot_wrap, 3)

        root.addLayout(mid, 1)

        # style local (si tu n’as pas QSS)
        self.setStyleSheet("""
            QGroupBox { font-weight: 700; }
            QFrame#MCPanel { background: white; border: 1px solid #cfcfcf; border-radius: 6px; }
            QFrame#MCCard { background: white; border: 1px solid #cfcfcf; border-radius: 6px; }
            QLabel#MCCardTitle { color: #0b6fa4; font-weight: 800; }
            QLabel#MCCardValue { font-size: 14px; font-weight: 700; }
        """)

    # ---------- Methods ----------
    def _mc_standard(self, S0, K, T, r, sigma, N, opt_type, seed=42):
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        Z = rng.standard_normal(N)
        ST = _simulate_ST(S0, T, r, sigma, Z)
        payoff = _payoff(ST, K, opt_type)
        disc = np.exp(-float(r) * float(T))
        samples = disc * payoff

        mean, var, _ = _mc_stats(samples)
        dt = time.perf_counter() - t0
        return mean, var, dt, samples

    def _mc_antithetic(self, S0, K, T, r, sigma, N, opt_type, seed=42):
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        if N % 2 != 0:
            N -= 1
        Z_half = rng.standard_normal(N // 2)
        Z = np.concatenate([Z_half, -Z_half])

        ST = _simulate_ST(S0, T, r, sigma, Z)
        payoff = _payoff(ST, K, opt_type)
        disc = np.exp(-float(r) * float(T))
        samples = disc * payoff

        # Les paires (Z, -Z) sont dépendantes : on calcule la variance sur les
        # moyennes de paires, puis on la ramène à une variance « par simulation »
        # (× 2) pour la comparer à la méthode standard à nombre de tirages égal.
        pairs = 0.5 * (samples[: N // 2] + samples[N // 2:])
        mean, var_pairs, _ = _mc_stats(pairs)
        var = 2.0 * var_pairs
        dt = time.perf_counter() - t0
        return mean, var, dt, samples


    def _mc_control_variate(self, S0, K, T, r, sigma, N, opt_type, seed=42):
        """
        Contrôle de variance avec X = discounted(ST) - E[discounted(ST)].
        Or E[e^{-rT} ST] = S0.
        """
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        Z = rng.standard_normal(N)
        ST = _simulate_ST(S0, T, r, sigma, Z)

        disc = np.exp(-float(r) * float(T))
        Y = disc * _payoff(ST, K, opt_type)          # target
        X = disc * ST - float(S0)                     # control with known mean 0

        # beta* = cov(Y,X)/var(X)
        vx = np.var(X, ddof=1)
        if vx <= 1e-18:
            Y_cv = Y
        else:
            beta = np.cov(Y, X, ddof=1)[0, 1] / vx
            Y_cv = Y - beta * X

        mean, var, _ = _mc_stats(Y_cv)
        dt = time.perf_counter() - t0
        return mean, var, dt, Y, Y_cv

    # ---------- UI ----------
    def _set_table(self, rows):
        self.table.setRowCount(len(rows))
        for i, (m, err, tsec, var) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(m)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{err:.6f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{tsec:.3f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{var:.6f}"))

    def on_run(self):
        try:
            S0 = float(self.in_S0.text())
            K = float(self.in_K.text())
            T = float(self.in_T.text())
            r = float(self.in_r.text())
            sigma = float(self.in_sigma.text())
            N = int(float(self.in_N.text()))
            opt_type = self.cmb_type.currentText().strip().lower()

            if S0 <= 0 or K <= 0 or T <= 0 or sigma < 0 or N < 100:
                raise ValueError("Contraintes: S0>0, K>0, T>0, sigma>=0, N>=100")

            # Prix exact (BS)
            bs = float(bs_price(S0, K, T, r, sigma, opt_type))

            # Standard
            p_std, v_std, t_std, samp_std = self._mc_standard(S0, K, T, r, sigma, N, opt_type, seed=42)
            err_std = abs(p_std - bs)

            # Antithetic
            p_ant, v_ant, t_ant, samp_ant = self._mc_antithetic(S0, K, T, r, sigma, N, opt_type, seed=42)
            err_ant = abs(p_ant - bs)

            # Control variate (améliorée)
            p_cv, v_cv, t_cv, samp_y, samp_cv = self._mc_control_variate(S0, K, T, r, sigma, N, opt_type, seed=42)
            err_cv = abs(p_cv - bs)

            # Cards (comme l’image)
            self.card_std._value_label.setText(f"Erreur : {err_std:.3f}    |    Temps : {t_std:.2f} s")
            self.card_imp._value_label.setText(f"Erreur : {err_cv:.3f}    |    Temps : {t_cv:.2f} s")

            # Table
            rows = [
                ("Standard", err_std, t_std, v_std),
                ("Antithétique", err_ant, t_ant, v_ant),
                ("Contrôle Variance", err_cv, t_cv, v_cv),
            ]
            self._set_table(rows)

            # Plot histogram (comme l’image : 2 distributions)
            self.ax.clear()
            self.ax.set_title("Distribution des Simulations")
            self.ax.set_xlabel("Valeur simulée")
            self.ax.set_ylabel("Densité")

            # Light theme again (ax.clear() reset)
            self.ax.set_facecolor("white")
            self.ax.grid(True, color="#d9d9d9", alpha=0.6)
            for spine in self.ax.spines.values():
                spine.set_color("#000000")
            self.ax.tick_params(colors="#000000")
            self.ax.title.set_color("#000000")
            self.ax.xaxis.label.set_color("#000000")
            self.ax.yaxis.label.set_color("#000000")

            # hist overlay: Standard vs Améliorée (CV)
            self.ax.hist(samp_std, bins=60, density=True, alpha=0.8, label="Monte-Carlo Standard")
            self.ax.hist(samp_ant, bins=60, density=True, alpha=0.6, label="Antithétique")
            self.ax.hist(samp_cv,  bins=60, density=True, alpha=0.6, label="Méthode Améliorée")

            self.ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
