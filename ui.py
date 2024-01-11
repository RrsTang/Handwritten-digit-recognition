import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QBrush, QLinearGradient, QColor
from PyQt5.QtCore import Qt, QRect
from recoginze import MyConvNet
import torch
import cv2
import numpy as np
from PIL import Image

model = MyConvNet()
model.load_state_dict(torch.load('mymodel1.pth'))

class Board(QWidget):
    def __init__(self):
        super(Board, self).__init__()
        self.resize(500, 350)
        self.move(100, 100)
        self.setWindowTitle("手写数字识别 21307272")
        self.setMouseTracking(False)
        self.pos = []
        self.setStyleSheet("background-color: white;")

        self.testButton = QPushButton(self)
        self.testButton.setText("识别")
        self.testButton.setParent(self)
        self.testButton.move(380, 130)
        self.testButton.clicked.connect(self.testButtonclicked)

        self.clearButton = QPushButton(self)
        self.clearButton.setText("清除")
        self.clearButton.setParent(self)
        self.clearButton.move(380, 230)
        self.clearButton.clicked.connect(self.clearButtonclicked)

        self.text = QLabel()
        self.text.setText("识别结果:")
        self.text.setParent(self)
        self.text.move(340, 50)

        self.answer = QLabel()
        self.answer.setParent(self)
        self.answer.move(420, 50)

        self.text1 = QLabel()
        self.text1.setText("可信度:")
        self.text1.setParent(self)
        self.text1.move(355, 80)

        self.prob_label = QLabel()
        self.prob_label.setFixedSize(60, 20)
        self.prob_label.setParent(self)
        self.prob_label.move(420, 77)

    def testButtonclicked(self):
        im = self.grab(rectangle=QRect(20, 20, 300, 300)).toImage()
        im = Image.fromqimage(im)
        im = im.convert('L').resize((28, 28))
        im = np.array(im)
        im = torch.tensor(im, dtype=torch.float32).reshape(1, 28, 28)
        im = 255.0 - im
        prob = model(im)
        ind = int(torch.argmax(prob))
        self.answer.setText(str(ind))
        self.prob_label.setText(str(float(torch.max(prob))))

    def clearButtonclicked(self):
        self.pos = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        gradient = QLinearGradient(0, 0, 0, 100)
        gradient.setColorAt(0.0, QColor(50, 50, 50))
        gradient.setColorAt(1.0, QColor(0, 0, 0))
        brush = QBrush(gradient)
        pen = QPen(Qt.black, 30, Qt.SolidLine, cap=Qt.RoundCap)
        pen.setBrush(brush)
        painter.drawRect(20, 20, 300, 300)
        painter.setPen(pen)
        if len(self.pos) > 1:
            start = self.pos[0]
            for tmp in self.pos:
                end = tmp
                if end == (-1, -1):
                    start = (-1, -1)
                    continue
                if start == (-1, -1):
                    start = end
                    continue
                painter.drawLine(start[0], start[1], end[0], end[1])
                start = end
        painter.end()

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if 10 <= x <= 310 and 10 <= y <= 310:
            tmp = (x, y)
            self.pos.append(tmp)
            self.update()

    def mouseReleaseEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if 10 <= x <= 310 and 10 <= y <= 310:
            tmp = (-1, -1)
            self.pos.append(tmp)
            self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    board = Board()
    board.show()
    sys.exit(app.exec_())