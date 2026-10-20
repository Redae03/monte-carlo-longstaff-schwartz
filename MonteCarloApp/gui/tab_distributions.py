import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QGroupBox, QMessageBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from MonteCarloApp.core.distributions import exponential_inversion, box_muller, uniform_ab


class DistributionsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        box = QGroupBox("Simulation de lois")
        grid = QGridLayout(box)

        self.cmb = QComboBox()
        self.cmb.addItems(["normale", "exponentielle", "uniforme"])

        self.in_p1 = QLineEdit("0")    
        self.in_p2 = QLineEdit("1")   
        self.in_N = QLineEdit("10000")

        grid.addWidget(QLabel("Loi"), 0, 0); grid.addWidget(self.cmb, 0, 1)
        grid.addWidget(QLabel("Param1"), 0, 2); grid.addWidget(self.in_p1, 0, 3)
        grid.addWidget(QLabel("Param2"), 1, 2); grid.addWidget(self.in_p2, 1, 3)
        grid.addWidget(QLabel("N"), 1, 0); grid.addWidget(self.in_N, 1, 1)

        self.btn = QPushButton("Simuler")
        self.btn.clicked.connect(self.on_simulate)
        grid.addWidget(self.btn, 2, 3)

        root.addWidget(box)

        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        root.addWidget(self.canvas)

        self.lbl_stats = QLabel("Stats: -")
        root.addWidget(self.lbl_stats)

    def on_simulate(self):
        try:
            law = self.cmb.currentText().strip().lower()
            p1 = float(self.in_p1.text())
            p2 = float(self.in_p2.text())
            N = int(float(self.in_N.text()))
            if N < 10:
                raise ValueError("N doit être >= 10")

            # ✅ reset la figure à chaque simulation (évite le bug des axes)
            self.fig.clear()

            # -------------------------
            # NORMALE (Box–Muller) -> 3 graphes + densité théorique
            # -------------------------
            if law == "normale":
                mu = p1
                sigma = abs(p2)
                if sigma == 0:
                    raise ValueError("Pour la normale : sigma doit être > 0")

                # Génération Box–Muller : Z1, Z2 ~ N(0,1)
                n_half = (N + 1) // 2
                Z1, Z2 = box_muller(n_half)

                # X1, X2 ~ N(mu, sigma^2)
                X1 = mu + sigma * Z1
                X2 = mu + sigma * Z2

                # Fusion (exactement N valeurs)
                X = np.concatenate([X1, X2])[:N]

                # Densité théorique N(mu, sigma^2)
                x_grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
                phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)

                ax1 = self.fig.add_subplot(131)
                ax2 = self.fig.add_subplot(132)
                ax3 = self.fig.add_subplot(133)

                ax1.hist(X1, bins=50, density=True, alpha=0.7)
                ax1.plot(x_grid, phi, "r", linewidth=2)
                ax1.set_title("Z1 (Box–Muller)")
                ax1.set_xlabel("x")
                ax1.set_ylabel("densité")
                ax1.grid(True, alpha=0.3)

                ax2.hist(X2, bins=50, density=True, alpha=0.7)
                ax2.plot(x_grid, phi, "r", linewidth=2)
                ax2.set_title("Z2 (Box–Muller)")
                ax2.set_xlabel("x")
                ax2.grid(True, alpha=0.3)

                ax3.hist(X, bins=50, density=True, alpha=0.7)
                ax3.plot(x_grid, phi, "r", linewidth=2)
                ax3.set_title("Z1 ∪ Z2 (fusion)")
                ax3.set_xlabel("x")
                ax3.grid(True, alpha=0.3)

                self.fig.tight_layout()
                self.canvas.draw()

                self.lbl_stats.setText(
                    f"Z1: mean={np.mean(X1):.6f}, var={np.var(X1, ddof=1):.6f} | "
                    f"Z2: mean={np.mean(X2):.6f}, var={np.var(X2, ddof=1):.6f} | "
                    f"Fusion: mean={np.mean(X):.6f}, var={np.var(X, ddof=1):.6f}"
                )
                return

            # -------------------------
            # EXPONENTIELLE
            # -------------------------
            elif law == "exponentielle":
                lamb = p1
                if lamb <= 0:
                    raise ValueError("Pour l'exponentielle : lambda doit être > 0")
                x = exponential_inversion(lamb, N)

            # -------------------------
            # UNIFORME
            # -------------------------
            else:  # uniforme
                a, b = p1, p2
                x = uniform_ab(a, b, N)

            # ✅ pour les autres lois : 1 seul graphe
            self.ax = self.fig.add_subplot(111)

            mean = float(np.mean(x))
            var = float(np.var(x, ddof=1))

            self.ax.set_title(f"Histogramme - {law}")
            self.ax.hist(x, bins=50, density=True, alpha=0.8)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("densité")
            self.ax.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw()

            self.lbl_stats.setText(f"Stats: mean={mean:.6f} | var={var:.6f}")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

