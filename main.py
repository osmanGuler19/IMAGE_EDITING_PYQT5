from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import  QMutex
from PyQt5.QtWidgets import QFileDialog, QPushButton, QMessageBox
from PyQt5.QtGui import QImage
from cv2 import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import  sato, unsharp_mask, prewitt
from skimage.exposure import adjust_gamma
from skimage import img_as_ubyte
from skimage.transform import swirl, warp
from skimage.util import img_as_float
import random


##UI maked with QT designer and I added some extra things via writing code
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 632)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 781, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.save = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save.sizePolicy().hasHeightForWidth())
        self.save.setSizePolicy(sizePolicy)
        self.save.setObjectName("save")
        self.horizontalLayout.addWidget(self.save)
        self.load = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.load.sizePolicy().hasHeightForWidth())
        self.load.setSizePolicy(sizePolicy)
        self.load.setObjectName("load")
        self.horizontalLayout.addWidget(self.load)
        self.reset = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.reset.sizePolicy().hasHeightForWidth())
        self.reset.setSizePolicy(sizePolicy)
        self.reset.setObjectName("reset")
        self.horizontalLayout.addWidget(self.reset)
        spacerItem = QtWidgets.QSpacerItem(150, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.horizontalSlider = QtWidgets.QSlider(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider.sizePolicy().hasHeightForWidth())
        self.horizontalSlider.setSizePolicy(sizePolicy)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout.addWidget(self.horizontalSlider)
        self.bulaniklik_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bulaniklik_label.sizePolicy().hasHeightForWidth())
        self.bulaniklik_label.setSizePolicy(sizePolicy)
        self.bulaniklik_label.setObjectName("bulaniklik_label")
        self.horizontalLayout.addWidget(self.bulaniklik_label)
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 50, 133, 411))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.goruntu_iyilestirme_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.goruntu_iyilestirme_label.setObjectName("goruntu_iyilestirme_label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.goruntu_iyilestirme_label)
        self.histogram = QtWidgets.QPushButton(self.formLayoutWidget)
        self.histogram.setIconSize(QtCore.QSize(10, 10))
        self.histogram.setObjectName("histogram")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.histogram)
        self.goruntu_iyiletirme = QtWidgets.QComboBox(self.formLayoutWidget)
        self.goruntu_iyiletirme.setObjectName("goruntu_iyiletirme")
        self.goruntu_iyiletirme.addItem("Gaussian")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.goruntu_iyiletirme.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.goruntu_iyiletirme)
        self.morfolojik_islemler_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.morfolojik_islemler_label.setObjectName("morfolojik_islemler_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.morfolojik_islemler_label)
        self.morfolojik_islemler = QtWidgets.QComboBox(self.formLayoutWidget)
        self.morfolojik_islemler.setObjectName("morfolojik_islemler")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.morfolojik_islemler.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.morfolojik_islemler)
        self.histogram_esitle = QtWidgets.QPushButton(self.formLayoutWidget)
        self.histogram_esitle.setObjectName("histogram_esitle")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.histogram_esitle)
        self.video = QtWidgets.QPushButton(self.formLayoutWidget)
        self.video.setObjectName("video")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.video)
        self.dondurme = QtWidgets.QPushButton(self.formLayoutWidget)
        self.dondurme.setObjectName("dondurme")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.dondurme)
        self.simetri = QtWidgets.QPushButton(self.formLayoutWidget)
        self.simetri.setObjectName("simetri")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.simetri)

        self.swirl_button = QtWidgets.QPushButton(self.formLayoutWidget)
        self.swirl_button.setObjectName("swirl_button")
        self.formLayout.setWidget(12, QtWidgets.QFormLayout.LabelRole, self.swirl_button)

        self.intensity_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.intensity_label.setObjectName("intensity_label")
        self.formLayout.setWidget(13, QtWidgets.QFormLayout.LabelRole, self.intensity_label)

        self.divider_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.intensity_label.setObjectName("divider_label")
        self.formLayout.setWidget(14, QtWidgets.QFormLayout.LabelRole, self.divider_label)

        self.gamma_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.intensity_label.setObjectName("gamma_label")
        self.formLayout.setWidget(15, QtWidgets.QFormLayout.LabelRole, self.gamma_label)

        self.gamma_edit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.formLayout.setWidget(16, QtWidgets.QFormLayout.LabelRole, self.gamma_edit)

        self.gain_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.intensity_label.setObjectName("gain_label")
        self.formLayout.setWidget(17, QtWidgets.QFormLayout.LabelRole, self.gain_label)

        self.gain_edit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.formLayout.setWidget(18, QtWidgets.QFormLayout.LabelRole, self.gain_edit)

        self.gamma_button = QtWidgets.QPushButton(self.formLayoutWidget)
        self.dondurme.setObjectName("gamma_button")
        self.formLayout.setWidget(19, QtWidgets.QFormLayout.LabelRole, self.gamma_button)

        self.goruntu = QtWidgets.QLabel(self.centralwidget)
        self.goruntu.setGeometry(QtCore.QRect(150, 50, 561, 411))
        self.goruntu.setAlignment(QtCore.Qt.AlignCenter)
        self.goruntu.setObjectName("goruntu")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(170, 460, 561, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.width_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.width_label.sizePolicy().hasHeightForWidth())
        self.width_label.setSizePolicy(sizePolicy)
        self.width_label.setObjectName("width_label")
        self.horizontalLayout_4.addWidget(self.width_label)
        self.width_edit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.width_edit.sizePolicy().hasHeightForWidth())
        self.width_edit.setSizePolicy(sizePolicy)
        self.width_edit.setObjectName("width_edit")
        self.horizontalLayout_4.addWidget(self.width_edit)
        self.height_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.height_label.sizePolicy().hasHeightForWidth())
        self.height_label.setSizePolicy(sizePolicy)
        self.height_label.setObjectName("height_label")
        self.horizontalLayout_4.addWidget(self.height_label)
        self.height_edit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.height_edit.sizePolicy().hasHeightForWidth())
        self.height_edit.setSizePolicy(sizePolicy)
        self.height_edit.setObjectName("height_edit")
        self.horizontalLayout_4.addWidget(self.height_edit)
        self.resize = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resize.sizePolicy().hasHeightForWidth())
        self.resize.setSizePolicy(sizePolicy)
        self.resize.setObjectName("resize")
        self.horizontalLayout_4.addWidget(self.resize)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(170, 500, 561, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.left_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_label.sizePolicy().hasHeightForWidth())
        self.left_label.setSizePolicy(sizePolicy)
        self.left_label.setObjectName("left_label")
        self.horizontalLayout_5.addWidget(self.left_label)
        self.left_edit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_edit.sizePolicy().hasHeightForWidth())
        self.left_edit.setSizePolicy(sizePolicy)
        self.left_edit.setObjectName("left_edit")
        self.horizontalLayout_5.addWidget(self.left_edit)
        self.right_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_label.sizePolicy().hasHeightForWidth())
        self.right_label.setSizePolicy(sizePolicy)
        self.right_label.setObjectName("right_label")
        self.horizontalLayout_5.addWidget(self.right_label)
        self.right_edit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_edit.sizePolicy().hasHeightForWidth())
        self.right_edit.setSizePolicy(sizePolicy)
        self.right_edit.setObjectName("right_edit")
        self.horizontalLayout_5.addWidget(self.right_edit)
        self.bottom_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bottom_label.sizePolicy().hasHeightForWidth())
        self.bottom_label.setSizePolicy(sizePolicy)
        self.bottom_label.setObjectName("bottom_label")
        self.horizontalLayout_5.addWidget(self.bottom_label)
        self.bottom_edit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bottom_edit.sizePolicy().hasHeightForWidth())
        self.bottom_edit.setSizePolicy(sizePolicy)
        self.bottom_edit.setObjectName("bottom_edit")
        self.horizontalLayout_5.addWidget(self.bottom_edit)
        self.top_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.top_label.sizePolicy().hasHeightForWidth())
        self.top_label.setSizePolicy(sizePolicy)
        self.top_label.setObjectName("top_label")
        self.horizontalLayout_5.addWidget(self.top_label)
        self.top_edit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.top_edit.sizePolicy().hasHeightForWidth())
        self.top_edit.setSizePolicy(sizePolicy)
        self.top_edit.setObjectName("top_edit")
        self.horizontalLayout_5.addWidget(self.top_edit)
        self.kirp_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.kirp_button.setObjectName("kirp_button")
        self.horizontalLayout_5.addWidget(self.kirp_button)
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setGeometry(QtCore.QRect(760, 110, 22, 161))
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(750, 80, 51, 20))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)





        self.load.clicked.connect(self.loadImage)
        self.save.clicked.connect(self.savePhoto)
        self.verticalSlider.valueChanged['int'].connect(self.brightness_value)
        self.horizontalSlider.valueChanged['int'].connect(self.blur_value)
        self.reset.clicked.connect(self.resetImage)
        self.histogram_esitle.clicked.connect(self.histogramEsitle)
        self.video.clicked.connect(self.videoFilter)
        self.histogram.clicked.connect(self.histogramGoster)
        self.resize.clicked.connect(self.resizeImage)
        self.dondurme.clicked.connect(self.imageRotate)
        self.simetri.clicked.connect(self.imageSymmetry)
        self.kirp_button.clicked.connect(self.imageCrop)
        self.goruntu_iyiletirme.activated.connect(self.goruntuIyilestirme)
        self.morfolojik_islemler.activated.connect(self.morfolojikIslemler)
        self.swirl_button.clicked.connect(self.makeSwirl)
        self.gamma_button.clicked.connect(self.gammaImage)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)





    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Gorüntü İsleme"))
        self.save.setText(_translate("MainWindow", "Save"))
        self.load.setText(_translate("MainWindow", "Load"))
        self.reset.setText(_translate("MainWindow", "Reset"))
        self.bulaniklik_label.setText(_translate("MainWindow", "Bulanıklık"))
        self.goruntu_iyilestirme_label.setText(_translate("MainWindow", "Gorüntü İyiestirme"))
        self.histogram.setText(_translate("MainWindow", "Histogram Gorüntüle"))
        self.goruntu_iyiletirme.setItemText(0, _translate("MainWindow", "Seciniz"))
        self.goruntu_iyiletirme.setItemText(1, _translate("MainWindow", "Gaussian"))
        self.goruntu_iyiletirme.setItemText(2, _translate("MainWindow", "Sobel"))
        self.goruntu_iyiletirme.setItemText(3, _translate("MainWindow", "Sato"))
        self.goruntu_iyiletirme.setItemText(4, _translate("MainWindow", "Bileteral"))
        self.goruntu_iyiletirme.setItemText(5, _translate("MainWindow", "Unsharp_Mask"))
        self.goruntu_iyiletirme.setItemText(6, _translate("MainWindow", "Prewitt"))
        self.goruntu_iyiletirme.setItemText(7, _translate("MainWindow", "Salt&Pepper"))
        self.goruntu_iyiletirme.setItemText(8, _translate("MainWindow", "Laplacian"))
        self.goruntu_iyiletirme.setItemText(9, _translate("MainWindow", "Box Filter"))
        self.goruntu_iyiletirme.setItemText(10, _translate("MainWindow", "Erode"))
        self.morfolojik_islemler_label.setText(_translate("MainWindow", "Morfolojik İslemler"))
        self.morfolojik_islemler.setItemText(0, _translate("MainWindow", "Seciniz"))
        self.morfolojik_islemler.setItemText(1, _translate("MainWindow", "Erosion"))
        self.morfolojik_islemler.setItemText(2, _translate("MainWindow", "Dilation"))
        self.morfolojik_islemler.setItemText(3, _translate("MainWindow", "Opening"))
        self.morfolojik_islemler.setItemText(4, _translate("MainWindow", "Closing"))
        self.morfolojik_islemler.setItemText(5, _translate("MainWindow", "Morphological Gradient"))
        self.morfolojik_islemler.setItemText(6, _translate("MainWindow", "Top Hat"))
        self.morfolojik_islemler.setItemText(7, _translate("MainWindow", "Black Hat"))
        self.morfolojik_islemler.setItemText(8, _translate("MainWindow", "Rect"))
        self.morfolojik_islemler.setItemText(9, _translate("MainWindow", "Cross"))
        self.morfolojik_islemler.setItemText(10, _translate("MainWindow", "Ellipse"))
        self.histogram_esitle.setText(_translate("MainWindow", "Histogram Esitle"))
        self.video.setText(_translate("MainWindow", "Video-Kenar Esitle"))
        self.dondurme.setText(_translate("MainWindow", "Dondürme"))
        self.simetri.setText(_translate("MainWindow", "Simetri (Dikey)"))
        self.swirl_button.setText(_translate("MainWindow", "Swirl"))
        self.intensity_label.setText(_translate("MainWindow", "Intensity Operations"))
        self.divider_label.setText(_translate("MainWindow","----- 0-1 arası float-----"))
        self.gamma_label.setText(_translate("MainWindow", "Gamma"))
        self.gain_label.setText(_translate("MainWindow","Gain"))
        self.gamma_button.setText(_translate("MainWindow","Adjust Gamma"))
        self.goruntu.setText(_translate("MainWindow", "Goruntu"))
        self.width_label.setText(_translate("MainWindow", "Width"))
        self.height_label.setText(_translate("MainWindow", "Height"))
        self.resize.setText(_translate("MainWindow", "Resize"))
        self.left_label.setText(_translate("MainWindow", "Left"))
        self.right_label.setText(_translate("MainWindow", "Right"))
        self.bottom_label.setText(_translate("MainWindow", "Bottom"))
        self.top_label.setText(_translate("MainWindow", "Top"))
        self.kirp_button.setText(_translate("MainWindow", "Crop"))
        self.label.setText(_translate("MainWindow", "Parlaklık"))


        ###### İmage Processing için kullanacağımız değişkenler
        self._mutex = QMutex()
        self.tempImg = None
        self.resize_width = 0
        self.resize_height = 0

        self.brightness_val = 0
        self.blur_val = 0

        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0
        self.w= 0
        self.h=0



    def loadImage(self):
        filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        if filename !="":
            self.image = cv2.imread(filename)
            self.pil_image = Image.open(filename)
            self.setPhoto(self.image)


    def setPhoto(self,image):
        self.tempImg=image
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        im = QtGui.QPixmap.fromImage(image)
        self.tempImg = image
        self.goruntu.setPixmap(im.scaled(self.goruntu.size(),QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def convertQImageToMat(self,incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''
        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr

    def showDialog(self,message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(message)
        msgBox.setWindowTitle("Dialog Message")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            print('OK clicked')

    # save method
    def savePhoto(self):
        if self.tempImg is not None:
            self._mutex.lock()
            cv2.imwrite("snapshot.jpg", self.convertQImageToMat(self.tempImg))
            self._mutex.unlock()
            self.showDialog("Resim kaydedildi")

        else:
            self.goruntu.setText('Kaydetmek icin once resim seciniz')


    def brightness_value(self,value):
        self.brightness_val= value

        self.update()

    def blur_value(self,value):
        self.blur_val = value
        self.update()

    def changeBrightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def changeBlur(self, img, value):
        kernel_size = (value + 1, value + 1)  # +1 is to avoid 0
        img = cv2.blur(img, kernel_size)
        return img

    def update(self):
        if self.tempImg is None:
            self.goruntu.setText('Lutfen Once Resim Seciniz')

        else:

            img = self.changeBrightness(self.image, self.brightness_val)
            img = self.changeBlur(img,self.blur_val)
            self.setPhoto(img)


    def resetImage(self):
        self.setPhoto(self.image)


    def histogramGoster(self):
        if self.tempImg is not None:
            img = cv2.cvtColor(self.convertQImageToMat(self.tempImg), cv2.COLOR_RGB2GRAY)
            histr = cv2.calcHist([img], [0], None, [256], [0, 256])
            # show the plotting graph of an image
            plt.plot(histr)
            plt.show()


    def histogramEsitle(self):
        if self.tempImg is not None:
            img = cv2.cvtColor(self.convertQImageToMat(self.tempImg), cv2.COLOR_RGB2GRAY)
            img_hist = cv2.equalizeHist(img)
            self.setPhoto(img_hist)


    def videoFilter(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()  # ret gets a boolean value.
            frame = cv2.GaussianBlur(frame, (7, 7), 1.41)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(frame, 15, 80)
            cv2.imshow('Canny Edge', edge)
            if cv2.waitKey(20) == ord('q'):  # Introduce 20 milisecond delay. press q to exit.
                break
        cv2.destroyAllWindows()


    def resizeImage(self):

        if self.tempImg is not None:
            self.resize_width = int(self.width_edit.text())
            self.resize_height = int(self.height_edit.text())
            dim = (self.resize_width, self.resize_height)
            self.tempImg = cv2.resize(self.convertQImageToMat(self.tempImg), dim, interpolation = cv2.INTER_AREA)
            self.setPhoto(self.tempImg)

    def imageRotate(self):
        if self.tempImg is not None:
            rotatedImg = cv2.rotate(self.convertQImageToMat(self.tempImg), cv2.ROTATE_90_CLOCKWISE)
            self.setPhoto(rotatedImg)


    def imageSymmetry(self):
            if self.tempImg is not None:
                flip = cv2.flip(self.convertQImageToMat(self.tempImg), 1)
                self.setPhoto(flip)


    def imageCrop(self):
        if self.tempImg is not None:
            if(self.left_edit.text() != "" and self.right_edit.text() != "" and self.bottom_edit.text() != "" and self.top_edit.text() != ""):
                self.left = int(self.left_edit.text())
                self.right = int(self.right_edit.text())
                self.top = int(self.top_edit.text())
                self.bottom = int(self.bottom_edit.text())
                try:
                    crop_img = self.convertQImageToMat(self.tempImg)[:, self.left:int(self.w)-self.right]
                    crop_img = crop_img[self.top:int(self.h) - self.bottom, :]
                    dim1 = len(crop_img)
                    dim2 = len(crop_img[0])
                    self.width = dim2
                    self.height = dim1
                    self.width = str(self.width)
                    self.height = str(self.height)
                    self.setPhoto(crop_img)
                except:
                    print("Kirpma hatasi!")

    def makeSwirl(self):
        if self.tempImg is not None:
            #print(type(self.tempImg))
            self._mutex.lock()
            cv_image = self.convertQImageToMat(self.tempImg)
            swirled = swirl(img_as_float(cv_image), rotation=0, strength=10, radius=int(self.tempImg.width()/3))
            cv_image = img_as_ubyte(swirled)
            self._mutex.unlock()
            self.setPhoto(cv_image)

    def add_noise(self,img):

        # Getting the dimensions of the image
        row, col = img.shape

        # Randomly pick some pixels in the
        # image for coloring them white
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)

            # Color that pixel to white
            img[y_coord][x_coord] = 255

        # Randomly pick some pixels in
        # the image for coloring them black
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)

            # Color that pixel to black
            img[y_coord][x_coord] = 0

        return img




    def laplacianFilter(self, img):
        s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = cv2.Laplacian(s, cv2.CV_16S, ksize=3)
        s = cv2.convertScaleAbs(s)
        return s

    def sobelFilter(self,img):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        src = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad




    def goruntuIyilestirme(self):
        filter_name = str(self.goruntu_iyiletirme.currentText())
        tmp_img2 = self.convertQImageToMat(self.tempImg)
        if self.tempImg is None:
            self.goruntu.setText('Goruntu iyilestirme filtreleri için once resim secin')
            return
        if filter_name.lower()=='seciniz':
            self.resetImage()
        elif filter_name.lower()=='gaussian':
            gaussianBlurKernel = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32) / 9
            gaussian_filtered = cv2.filter2D(src=tmp_img2, kernel=gaussianBlurKernel, ddepth=-1)
            self.setPhoto(gaussian_filtered)
        elif filter_name.lower()=='sobel':
            filtered_image = self.sobelFilter(self.image)
            self.setPhoto(filtered_image)

        elif filter_name.lower()=='sato':
            cv_image = tmp_img2
            sato_filtered = sato(img_as_float(cv_image), black_ridges=True, mode='wrap')
            cv_image = img_as_ubyte(sato_filtered)
            self.setPhoto(cv_image)

        elif filter_name.lower()=='bileteral':
            bileteral_filtered = cv2.bilateralFilter(self.image, 15, 75, 75)
            self.setPhoto(bileteral_filtered)

        elif filter_name.lower()=='unsharp_mask':
            cv_image = tmp_img2
            unsharp_masked = unsharp_mask(img_as_float(cv_image))
            cv_image = img_as_ubyte(unsharp_masked)
            self.setPhoto(cv_image)

        elif filter_name.lower()=='prewitt':
            prewitt_masked = prewitt(self.image)
            cv_image = img_as_ubyte(prewitt_masked)
            self.setPhoto(cv_image)

        elif filter_name.lower()=='salt&pepper':
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            filtered_image = self.add_noise(gray_image)
            self.setPhoto(filtered_image)

        elif filter_name.lower()=='laplacian':
            filtered_img = self.laplacianFilter(self.image)
            self.setPhoto(filtered_img)

        elif filter_name.lower()=='box filter':
            filtered_img =  cv2.boxFilter(self.image, -1, (31, 31))
            self.setPhoto(filtered_img)

        elif filter_name.lower()=='erode':
            # Creating kernel
            kernel = np.ones((5, 5), np.uint8)
            # Using cv2.erode() method
            image = cv2.erode(self.image, kernel)
            self.setPhoto(image)

    def gammaImage(self):
        if float(self.gain_edit.text()) < -1.0 and float(self.gain_edit.text()) > 1.0:
            self.showDialog("Gain should be in between -1 and 1")
            return
        elif self.tempImg is not None and self.gain_edit.text()!="" and self.gamma_edit.text()!="":
            try:
                cv_image = self.convertQImageToMat(self.tempImg)
                gamma_filtered = adjust_gamma(img_as_float(cv_image), gamma= float(self.gamma_edit.text()), gain= float(self.gain_edit.text()))
                cv_image = img_as_ubyte(gamma_filtered)
                self.setPhoto(cv_image)
            except:
                self.showDialog("Gamma işlemi yapılırken hata meydana geldi. \n 0 ve 1 arasında değer vermeyi deneyin.")
        else:
            self.showDialog("Bilinmeyen Hata!")
    def morfolojikIslemler(self):
        filter_name = str(self.morfolojik_islemler.currentText())
        kernel = np.ones((5, 5), np.uint8)

        if filter_name.lower()== "erosion":
            erosion_filtered = cv2.erode(self.image,kernel)
            self.setPhoto(erosion_filtered)
        elif filter_name.lower()== "dilation":
            erosion_filtered = cv2.dilate(self.image,kernel,iterations=1)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "opening":
            erosion_filtered = cv2.morphologyEx(self.image,cv2.MORPH_OPEN,kernel)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "closing":
            erosion_filtered = cv2.dilate(self.image,kernel,iterations=1)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "morphological gradient":
            erosion_filtered = cv2.morphologyEx(self.image,cv2.MORPH_CLOSE,kernel)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "top hat":
            erosion_filtered = cv2.morphologyEx(self.image,cv2.MORPH_TOPHAT,kernel)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "black hat":
            erosion_filtered = cv2.morphologyEx(self.image,cv2.MORPH_BLACKHAT,kernel)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "rect":
            erosion_filtered = cv2.morphologyEx(self.image,cv2.MORPH_RECT,kernel)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "cross":
            erosion_filtered = cv2.morphologyEx(self.image,cv2.MORPH_CROSS,kernel)
            self.setPhoto(erosion_filtered)

        elif filter_name.lower()== "ellipse":
            erosion_filtered = cv2.morphologyEx(self.image,cv2.MORPH_ELLIPSE,kernel,)
            self.setPhoto(erosion_filtered)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
