import sys, os
from PyQt5.QtWidgets import QApplication
from MonteCarloApp.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qss_path = os.path.join(os.path.dirname(__file__), "gui", "styles.qss")

    try:
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass


    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

