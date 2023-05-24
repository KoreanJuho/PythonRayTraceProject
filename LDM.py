import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication

mypath = r"./LDM.ui"
form_window = uic.loadUiType(mypath)[0]

class MainWindow(QMainWindow, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()