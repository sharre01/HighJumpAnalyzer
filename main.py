import sys
from PyQt5.QtWidgets import QApplication
from gui import SportsAnalysisApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SportsAnalysisApp()
    ex.show()

    with open("style.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    sys.exit(app.exec_())
