import numpy as np
import time

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QMessageBox, QAbstractItemView, QFrame
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from MonteCarloApp.core.options import american_option_lsm, lsm_immediate_vs_continuation


def wrap_card(widget):
    card = QFrame()
    card.setFrameShape(QFrame.StyledPanel)
    card.setStyleSheet("QFrame{background:white; border:1px solid #cfcfcf; border-radius:6px;}")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.addWidget(widget)
    return card


class OptionsTab(QWidget):
    """
    Onglet: Option Américaine (Longstaff–Schwartz / LSM)
    - Call / Put
    - Retourne un prix LSM
    - Affiche un petit résumé dans la table + texte dans le plot
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ------------------ Paramètres LSM ------------------
        box_params = QGroupBox("Paramètres (Option Américaine - LSM)")
        grid = QGridLayout(box_params)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems(["put", "call"])  # ✅ put aussi

        self.in_S0 = QLineEdit("100")
        self.in_K = QLineEdit("105")
        self.in_T = QLineEdit("1")
        self.in_r = QLineEdit("0.05")
        self.in_sigma = QLineEdit("0.2")

        self.in_steps = QLineEdit("50")
        self.in_paths = QLineEdit("20000")
        self.in_degree = QLineEdit("2")

        row = 0
        grid.addWidget(QLabel("Type"), row, 0); grid.addWidget(self.cmb_type, row, 1)
        grid.addWidget(QLabel("S0"),   row, 2); grid.addWidget(self.in_S0, row, 3)
        row += 1
        grid.addWidget(QLabel("K"),          row, 0); grid.addWidget(self.in_K, row, 1)
        grid.addWidget(QLabel("T (années)"), row, 2); grid.addWidget(self.in_T, row, 3)
        row += 1
        grid.addWidget(QLabel("r"),     row, 0); grid.addWidget(self.in_r, row, 1)
        grid.addWidget(QLabel("sigma"), row, 2); grid.addWidget(self.in_sigma, row, 3)
        row += 1
        grid.addWidget(QLabel("n_steps"), row, 0); grid.addWidget(self.in_steps, row, 1)
        grid.addWidget(QLabel("n_paths"), row, 2); grid.addWidget(self.in_paths, row, 3)
        row += 1
        grid.addWidget(QLabel("degree"), row, 0); grid.addWidget(self.in_degree, row, 1)

        self.btn_run = QPushButton("Calculer (LSM)")
        self.btn_run.clicked.connect(self.on_run_clicked)
        grid.addWidget(self.btn_run, row, 3)

        root.addWidget(box_params)

        # ------------------ Table + Plot ------------------
        mid = QHBoxLayout()
        mid.setSpacing(14)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Type", "Prix LSM", "Temps (ms)", "Paramètres"
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        from PyQt5.QtWidgets import QHeaderView
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Plot (juste du texte pour l’instant)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.fig.tight_layout()

        mid.addWidget(wrap_card(self.table), 2)
        mid.addWidget(wrap_card(self.canvas), 4)

        root.addLayout(mid)

    def _get_params(self):
        try:
            opt_type = self.cmb_type.currentText().strip().lower()

            S0 = float(self.in_S0.text())
            K = float(self.in_K.text())
            T = float(self.in_T.text())
            r = float(self.in_r.text())
            sigma = float(self.in_sigma.text())

            n_steps = int(float(self.in_steps.text()))
            n_paths = int(float(self.in_paths.text()))
            degree = int(float(self.in_degree.text()))
        except Exception:
            raise ValueError("Paramètres invalides: vérifie que tous les champs sont numériques.")

        if S0 <= 0 or K <= 0 or T <= 0 or sigma < 0:
            raise ValueError("Contraintes: S0>0, K>0, T>0, sigma>=0.")
        if n_steps < 1 or n_paths < 2 or degree < 1:
            raise ValueError("Contraintes LSM: n_steps>=1, n_paths>=2, degree>=1.")
        if opt_type not in ("call", "put"):
            raise ValueError("Type doit être call ou put.")

        return S0, K, T, r, sigma, n_steps, n_paths, degree, opt_type

    def _set_table(self, opt_type, price, t_ms, params_txt):
        self.table.setRowCount(1)
        self.table.setItem(0, 0, QTableWidgetItem(opt_type))
        self.table.setItem(0, 1, QTableWidgetItem(f"{price:.6f}"))
        self.table.setItem(0, 2, QTableWidgetItem(f"{t_ms:.2f}"))
        self.table.setItem(0, 3, QTableWidgetItem(params_txt))

    def on_run_clicked(self):
        try:
            S0, K, T, r, sigma, n_steps, n_paths, degree, opt_type = self._get_params()

            t0 = time.perf_counter()
            price, diag = american_option_lsm(
                S0, K, T, r, sigma,
                n_steps=n_steps,
                n_paths=n_paths,
                option_type=opt_type,
                degree=degree,
                return_diagnostics=True
            )
            t_ms = (time.perf_counter() - t0) * 1000.0

            params_txt = f"steps={n_steps}, paths={n_paths}, degree={degree}"
            self._set_table(opt_type, float(price), t_ms, params_txt)

            # Affichage dans le plot
            # ===== Comparaison Payoff immédiat vs Valeur de continuation (LSM) =====
            self.ax.clear()

            # Choix d'un pas de temps (milieu de la maturité)
            t_idx = n_steps // 2

            St, imm, cont, exercise = lsm_immediate_vs_continuation(
                diag=diag,
                K=K,
                option_type=opt_type,
                t_index=t_idx,
                degree=degree
            )

            self.ax.set_title(
                f"LSM – Payoff immédiat vs continuation\n(t = {t_idx}/{n_steps})"
            )
            self.ax.set_xlabel("S(t)")
            self.ax.set_ylabel("Valeur")

            # Continue
            mask_cont = (imm > 0) & (~exercise)
            self.ax.scatter(
                St[mask_cont], imm[mask_cont],
                s=10, alpha=0.6, label="Payoff immédiat (continue)"
            )
            self.ax.scatter(
                St[mask_cont], cont[mask_cont],
                s=10, alpha=0.6, label="Continuation estimée"
            )

            # Exercise
            mask_ex = exercise
            self.ax.scatter(
                St[mask_ex], imm[mask_ex],
                s=14, alpha=0.9, label="Exercise (payoff > continuation)"
            )

            # Courbes lissées (triées par S)
            order = np.argsort(St)
            self.ax.plot(St[order], imm[order], linewidth=1.5, label="Payoff (trié)")

            order2 = np.argsort(St[mask_cont])
            self.ax.plot(
                St[mask_cont][order2],
                cont[mask_cont][order2],
                linewidth=1.5,
                label="Continuation (triée)"
            )

            self.ax.grid(True, alpha=0.35)
            self.ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()


        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
