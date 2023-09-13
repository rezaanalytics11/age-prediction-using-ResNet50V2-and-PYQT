from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import *
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QFormLayout
from PyQt5.QtWidgets import (QApplication, QWidget,
  QPushButton, QVBoxLayout, QHBoxLayout,QGridLayout,QLineEdit)
import cv2
from tensorflow import keras
from keras.preprocessing.image import img_to_array
import numpy as np


width=height=48

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        hbox0 = QHBoxLayout()
        self.label0 = QLabel('Please enter the picture URL and press the key', self)
        hbox0.addWidget(self.label0)

        self.file_name = QLineEdit(self)
        hbox0.addWidget(self.file_name)


        hbox1 = QHBoxLayout()

        self.label1 = QLabel('', self)
        hbox1.addWidget(self.label1)

        self.label3 = QLabel('', self)
        hbox1.addWidget(self.label3)

        hbox2 = QHBoxLayout()
        self.label4 = QLabel('', self)
        hbox2.addWidget(self.label4)

        Button = QPushButton('Key')
        hbox0.addWidget(Button)
        Button.clicked.connect(self.addurl)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox0)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.setLayout(vbox)
        #self.setGeometry(400, 400, 300, 150)
        self.setWindowTitle('Box layout example, QHBoxLayout, QVBoxLayout')
        self.show()

    def addurl(self):

        a=self.file_name.text()
        self.draw(a)

    def draw(self,file_name):
        new_model = keras.models.load_model(r'C:\Users\Ariya Rayaneh\Desktop\my_model_new.h5')
        image = cv2.imread(rf'C:\Users\Ariya Rayaneh\Desktop\{file_name}.jpg')

        img = img_to_array(image, data_format='channels_first')
        cv2.imwrite(r"C:\Users\Ariya Rayaneh\Desktop\human20_output.jpg", image)
        self.label1.setPixmap(QPixmap(r"C:\Users\Ariya Rayaneh\Desktop\human20_output.jpg"))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = image / 255.0
        image = image[np.newaxis, ...]
        result = new_model.predict(image)
        print(int(result[0][0]))

        self.label3.setText(str(int(result[0][0])))

if __name__ == '__main__':
 app = QApplication(sys.argv)
 ex = Example()
 sys.exit(app.exec_())
