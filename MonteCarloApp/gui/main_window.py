from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
)

from MonteCarloApp.gui.tab_distributions import DistributionsTab
from MonteCarloApp.gui.tab_processes import ProcessesTab
from MonteCarloApp.gui.tab_options import OptionsTab
from MonteCarloApp.gui.tab_methods import MethodsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulations Monte-Carlo")
        self.resize(1100, 650)

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(DistributionsTab(), "Simulations de Lois")
        tabs.addTab(ProcessesTab(), "Processus Stochastiques")
        tabs.addTab(MethodsTab(), "Options Européennes")
        tabs.addTab(OptionsTab(), "Options Américaines (LSM)")

