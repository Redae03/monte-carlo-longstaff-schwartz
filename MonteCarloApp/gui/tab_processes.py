import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QGroupBox, QMessageBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from MonteCarloApp.core.brownien import brownien_motion
from MonteCarloApp.core.black_scholes import gbm_paths

class ProcessesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        box = QGroupBox("Processus stochastiques")
        grid = QGridLayout(box)

        self.cmb = QComboBox()
        self.cmb.addItems([
            "Brownien (Euler)",
            "Black-Scholes (Euler)",
            "Black-Scholes (méthode directe)"
        ])


        self.in_S0 = QLineEdit("100")
        self.in_mu = QLineEdit("0.05")
        self.in_sigma = QLineEdit("0.2")
        self.in_T = QLineEdit("1")
        self.in_steps = QLineEdit("252")
        self.in_paths = QLineEdit("20")

        grid.addWidget(QLabel("Processus"), 0, 0); grid.addWidget(self.cmb, 0, 1)
        grid.addWidget(QLabel("S0"), 0, 2); grid.addWidget(self.in_S0, 0, 3)
        grid.addWidget(QLabel("mu (drift)"), 1, 0); grid.addWidget(self.in_mu, 1, 1)
        grid.addWidget(QLabel("sigma"), 1, 2); grid.addWidget(self.in_sigma, 1, 3)
        grid.addWidget(QLabel("T"), 2, 0); grid.addWidget(self.in_T, 2, 1)
        grid.addWidget(QLabel("steps"), 2, 2); grid.addWidget(self.in_steps, 2, 3)
        grid.addWidget(QLabel("paths"), 3, 0); grid.addWidget(self.in_paths, 3, 1)

        self.btn = QPushButton("Simuler")
        self.btn.clicked.connect(self.on_simulate)
        grid.addWidget(self.btn, 3, 3)

        root.addWidget(box)

        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        bg = "#f2f2f2"
        gridc = "#d9d9d9"
        txt = "#000000"

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor("white")   # plot blanc, comme l'image
        for spine in self.ax.spines.values():
            spine.set_color("#000000")

        self.ax.tick_params(colors=txt)
        self.ax.xaxis.label.set_color(txt)
        self.ax.yaxis.label.set_color(txt)
        self.ax.title.set_color(txt)
        self.ax.grid(True, color=gridc, alpha=0.6)
        self.fig.tight_layout()
        root.addWidget(self.canvas)

    def on_simulate(self):
        try:
            proc = self.cmb.currentText().strip().lower()
            S0 = float(self.in_S0.text())
            mu = float(self.in_mu.text())
            sigma = float(self.in_sigma.text())
            T = float(self.in_T.text())
            n_steps = int(float(self.in_steps.text()))
            n_paths = int(float(self.in_paths.text()))

            if T <= 0 or n_steps < 1 or n_paths < 1:
                raise ValueError("Contraintes: T>0, steps>=1, paths>=1")

            self.ax.clear()

            # temps (utile pour brownien + Euler GBM)
            t = np.linspace(0, T, n_steps + 1)

            if proc == "brownien":
                self.ax.set_title("Brownien W(t)")
                self.ax.set_xlabel("t")
                self.ax.set_ylabel("W(t)")

                for _ in range(n_paths):
                    W = brownien_motion(T, n_steps)
                    self.ax.plot(t, W)

            elif proc == "black-scholes (méthode directe)":
                # Ici : on simule S_T directement (pas de trajectoire)
                self.ax.set_title("Black–Scholes – méthode directe")
                self.ax.set_xlabel("S_T")
                self.ax.set_ylabel("densité")

                Z = np.random.randn(n_paths)
                ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

                self.ax.hist(ST, bins=50, density=True, alpha=0.7)

            else:
                # Euler GBM (trajectoires)
                self.ax.set_title("GBM S(t) (Black-Scholes) – Euler")
                self.ax.set_xlabel("t")
                self.ax.set_ylabel("S(t)")

                paths = gbm_paths(S0, T, mu, sigma, n_steps, n_paths)
                for j in range(n_paths):
                    self.ax.plot(t, paths[:, j])

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

