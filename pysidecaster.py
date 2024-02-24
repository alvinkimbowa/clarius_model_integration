#!/usr/bin/env python

import ctypes
import os.path
import sys
import cv2
import torch
import time
from datetime import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Final
from PIL.ImageQt import ImageQt

from torchvision.transforms import ToTensor

from nnunet import NNUNet

if sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle  # load the libcast.so shared library
    pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")  # load the pyclariuscast.so shared library

import pyclariuscast
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot

CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14


# custom event for handling change in freeze state
class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen


# custom event for handling button presses
class ButtonEvent(QtCore.QEvent):
    def __init__(self, btn, clicks):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks


# custom event for handling new images
class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))


# manages custom events posted from callbacks, then relays as signals to the main widget
class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.usimage = QtGui.QImage()

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        return True


# global required for the cast api callbacks
signaller = Signaller()


# draws the ultrasound image
class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast):
        QtWidgets.QGraphicsView.__init__(self)
        self.cast = cast
        self.setScene(QtWidgets.QGraphicsScene())

    # set the new image and redraw
    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()

    # saves a local image
    def saveImage(self):
        self.image.save(str(Path.home() / "Pictures/clarius_image.png"))

    # resize the scan converter, image, and scene
    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        self.cast.setOutputSize(w, h)
        self.image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.black)
        self.setSceneRect(0, 0, w, h)

    # black background
    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    # draws the image
    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)


# main widget with controls and ui
class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.cast = cast
        self.setWindowTitle("Clarius Cast Demo")

        # create central widget within main window
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        ip = QtWidgets.QLineEdit("192.168.1.1")
        ip.setInputMask("000.000.000.000")
        port = QtWidgets.QLineEdit("5828")
        port.setInputMask("00000")

        conn = QtWidgets.QPushButton("Connect")
        self.run = QtWidgets.QPushButton("Run")
        quit = QtWidgets.QPushButton("Quit")
        depthUp = QtWidgets.QPushButton("< Depth")
        depthDown = QtWidgets.QPushButton("> Depth")
        gainInc = QtWidgets.QPushButton("> Gain")
        gainDec = QtWidgets.QPushButton("< Gain")
        # captureImage = QtWidgets.QPushButton("Capture Image")
        captureImage = QtWidgets.QPushButton("Save local")
        captureCine = QtWidgets.QPushButton("Capture Movie")
        saveImage = QtWidgets.QPushButton("Next Patient")
        bMode = QtWidgets.QPushButton("B Mode")
        cfiMode = QtWidgets.QPushButton("Color Mode")

        # try to connect/disconnect to/from the probe
        def tryConnect():
            if not cast.isConnected():
                if cast.connect(ip.text(), int(port.text()), "research"):
                    self.statusBar().showMessage("Connected")
                    conn.setText("Disconnect")
                else:
                    self.statusBar().showMessage("Failed to connect to {0}".format(ip.text()))
            else:
                if cast.disconnect():
                    self.statusBar().showMessage("Disconnected")
                    conn.setText("Connect")
                else:
                    self.statusBar().showMessage("Failed to disconnect")

        # try to freeze/unfreeze
        def tryFreeze():
            if cast.isConnected():
                cast.userFunction(CMD_FREEZE, 0)

        # try depth up
        def tryDepthUp():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_DEC, 0)

        # try depth down
        def tryDepthDown():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_INC, 0)

        # try gain down
        def tryGainDec():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_DEC, 0)

        # try gain up
        def tryGainInc():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_INC, 0)

        # try capture image
        def tryCaptureImage():
            # if cast.isConnected():
            #     cast.userFunction(CMD_CAPTURE_IMAGE, 0)
            global save_local
            save_local = not save_local # toggle the save_local variable

        # try capture cine
        def tryCaptureCine():
            if cast.isConnected():
                cast.userFunction(CMD_CAPTURE_CINE, 0)

        # try to save a local image
        def trySaveImage():
            # self.img.saveImage()            
            global new_subfolder_path
            new_subfolder_path = create_daily_folder(base_folder)

            global save_local
            if save_local:
                save_local = False # everytime a next patient is selected, you need to click save local again

        # try b mode
        def tryBMode():
            if cast.isConnected():
                cast.userFunction(CMD_B_MODE, 0)

        # try cfi mode
        def tryCfiMode():
            if cast.isConnected():
                cast.userFunction(CMD_CFI_MODE, 0)
        

        conn.clicked.connect(tryConnect)
        self.run.clicked.connect(tryFreeze)
        quit.clicked.connect(self.shutdown)
        depthUp.clicked.connect(tryDepthUp)
        depthDown.clicked.connect(tryDepthDown)
        gainInc.clicked.connect(tryGainInc)
        gainDec.clicked.connect(tryGainDec)
        captureImage.clicked.connect(tryCaptureImage)
        captureCine.clicked.connect(tryCaptureCine)
        saveImage.clicked.connect(trySaveImage)
        bMode.clicked.connect(tryBMode)
        cfiMode.clicked.connect(tryCfiMode)

        # add widgets to layout
        self.img = ImageView(cast)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img)

        inplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(inplayout)
        inplayout.addWidget(ip)
        inplayout.addWidget(port)

        connlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(connlayout)
        connlayout.addWidget(conn)
        connlayout.addWidget(self.run)
        connlayout.addWidget(quit)
        central.setLayout(layout)

        prmlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(prmlayout)
        prmlayout.addWidget(depthUp)
        prmlayout.addWidget(depthDown)
        prmlayout.addWidget(gainDec)
        prmlayout.addWidget(gainInc)

        caplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(caplayout)
        caplayout.addWidget(captureImage)
        caplayout.addWidget(captureCine)
        caplayout.addWidget(saveImage)
        caplayout.addWidget(bMode)

        modelayout = QtWidgets.QHBoxLayout()
        layout.addLayout(modelayout)
        modelayout.addWidget(bMode)
        modelayout.addWidget(cfiMode)

        # connect signals
        signaller.freeze.connect(self.freeze)
        signaller.button.connect(self.button)
        signaller.image.connect(self.image)

        # get home path
        path = os.path.expanduser("~/")
        if cast.init(path, 640, 480):
            self.statusBar().showMessage("Initialized")
        else:
            self.statusBar().showMessage("Failed to initialize")

    # handles freeze messages
    @Slot(bool)
    def freeze(self, frozen):
        if frozen:
            self.run.setText("Run")
            self.statusBar().showMessage("Image Stopped")
        else:
            self.run.setText("Freeze")
            self.statusBar().showMessage("Image Running (check firewall settings if no image seen)")

    # handles button messages
    @Slot(int, int)
    def button(self, btn, clicks):
        self.statusBar().showMessage("Button {0} pressed w/ {1} clicks".format(btn, clicks))

    # handles new images
    @Slot(QtGui.QImage)
    def image(self, img):
        self.img.updateImage(img)

    # handles shutdown
    @Slot()
    def shutdown(self):
        if sys.platform.startswith("linux"):
            # unload the shared library before destroying the cast object
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
        self.cast.destroy()
        QtWidgets.QApplication.quit()


def save_image(image, folder_path, suffix=''):
    '''Image should be a PIL image'''
    # Save the image to the given folder
    today_date = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    dst = os.path.join(folder_path, f"{today_date}_{suffix}.png")
    image.save(dst)
    print(f"image saved to {dst}")


IMG_SIZE = (512,512)
def prediction(image, width, height, img_size=IMG_SIZE):
    # Preprocessing image for inference
    print('Prediction called')
    img = Image.frombytes('RGBA', (width, height), image)
    print("Image read")
    img = img.convert('L')
    image_input = img.resize((img_size[0], img_size[1]))
    # print("Image resized")
    if save_local:
        save_image(img, new_subfolder_path, 'image')

    # image = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)
    # print("image read!")

    # image_input = torch.from_numpy(image)
    image_input = ToTensor()(image_input)
    print("image_input: ", image_input.shape)
    image_input = image_input.to(dtype=torch.float32)
    image_input = torch.reshape(image_input, shape=(1,*image_input.shape))
    print("image_input: ", image_input.shape)
    # image_input = normalize(image_input, axis=1)

    start_time = time.time()
    pred = model(image_input)
    pred = pred.detach().numpy()[0]
    pred = np.argmax(pred, 0).astype(np.uint8)
    # pred = (pred > 0.1).astype(np.uint8)
    inf_time = time.time() - start_time
    print("pred shape: ", pred.shape)
    print("pred: ", np.unique(pred), type(pred))
    print("inference_time: ", inf_time)

    # Use PIL to resize as it utilizes anti-aliasing
    pred = Image.fromarray(pred, mode='L').resize((width, height))
    # Resizing with anti aliasing introduces values between 0 and 1
    pred = (np.asarray(pred)>0.1).astype(np.uint8)
    
    if save_local:
        save_image(Image.fromarray(pred*255, mode='L'), new_subfolder_path, 'segmentation')

    # Augmentation
    print("Drawing predictions")
    start_time = time.time()
    img = np.asarray(img)
    print("Finding contours")
    contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Found contours")
    image_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = image_out.copy()
    print("Created canvas")
    cv2.drawContours(image_out, contours, -1, (255, 255, 0), thickness=cv2.FILLED)
    alpha = 0.25
    cv2.addWeighted(image_out, alpha, overlay, 1-alpha, 0, image_out)
    draw_time = time.time() - start_time
    print("Finished drawing predictions!")
    print("Post processing time: ", draw_time)

    # Convert augmented image to QImage
    # width = image.shape
    # height =
    # image = QtGui.QImage(image.data, image.shape[0],image.shape[1], QtGui.QImage.Format_RGB32)
    # image_out = Image.fromarray(np.uint8(image_out)).convert('RGB')
    image_out = Image.fromarray(np.uint8(image_out))
    if save_local:
        save_image(image_out, new_subfolder_path, 'image_segmentation_overlay')
    return ImageQt(image_out)


def get_last_assigned_number(folder_path):
    # Get a list of subfolders in the given folder
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # If there are no subfolders, return 0
    if not subfolders:
        return 0

    # Extract the numbers from the subfolder names and find the maximum
    numbers = [int(subfolder.split('_')[-1]) for subfolder in subfolders]
    return max(numbers)

def create_daily_folder(base_path, prefix='Patient'):
    # Create a folder with the current date
    today_date = datetime.now().strftime("%Y-%m-%d")
    daily_folder = os.path.join(base_path, today_date)
    os.makedirs(daily_folder, exist_ok=True)

    # Get the last assigned number for today's folder
    last_assigned_number = get_last_assigned_number(daily_folder)

    # Autogenerate the next number
    new_number = last_assigned_number + 1

    # Create a subfolder with the autogenerated number
    subfolder_name = f"{prefix}_{new_number}"
    subfolder_path = os.path.join(daily_folder, subfolder_name)
    os.makedirs(subfolder_path)

    return subfolder_path

count = 0
save_local = False
base_folder = "recorded_data"
new_subfolder_path = create_daily_folder(base_folder, prefix='Dummy')

## called when a new processed image is streamed
# @param image the scan-converted image data
# @param width width of the image in pixels
# @param height height of the image in pixels
# @param sz full size of image
# @param micronsPerPixel microns per pixel
# @param timestamp the image timestamp in nanoseconds
# @param angle acquisition angle for volumetric data
# @param imu inertial data tagged with the frame
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    # bpp = sz / (width * height)
    # print("bpp: ", bpp)
    # if bpp == 4:
    #     img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    # else:
    #     img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    
    ##### This is code to get prediction
    print("Inside newProcessedIMage...")
    # print("width: ", width)
    # print("height: ", height)
    # print("sz:, ", sz)
    global count
    print("count: ", count)
    # count += 1
    if count==0:
        bpp = sz / (width * height)
        print("bpp: ", bpp)
        if bpp == 4:
            img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
            count += 1
        else:
            img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    elif count==30:
        img = prediction(image, width, height)
        count = 1
    else:
        count += 1
        return
    #####    
    # a deep copy is important here, as the memory from 'image' won't be valid after the event posting
    signaller.usimage = img.copy()
    evt = ImageEvent()
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


## called when a new raw image is streamed
# @param image the raw pre scan-converted image data, uncompressed 8-bit or jpeg compressed
# @param lines number of lines in the data
# @param samples number of samples in the data
# @param bps bits per sample
# @param axial microns per sample
# @param lateral microns per line
# @param timestamp the image timestamp in nanoseconds
# @param jpg jpeg compression size if the data is in jpeg format
# @param rf flag for if the image received is radiofrequency data
# @param angle acquisition angle for volumetric data
def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    # print("inside newRawImage")
    # # print("image: ", image)
    # print("lines: ", lines)
    # print("samples: ", samples)
    # print("bps: ", bps)
    # print("axial: ", axial)
    # print("lateral: ", lateral)
    # print("timestamp: ", timestamp)
    # print("jpg: ", jpg)
    # print("rf: ", rf)
    # print("angle: ", angle)
    return


## called when a new spectrum image is streamed
# @param image the spectral image
# @param lines number of lines in the spectrum
# @param samples number of samples per line
# @param bps bits per sample
# @param period line repetition period of spectrum
# @param micronsPerSample microns per sample for an m spectrum
# @param velocityPerSample velocity per sample for a pw spectrum
# @param pw flag that is true for a pw spectrum, false for an m spectrum
def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


## called when freeze state changes
# @param frozen the freeze state
def freezeFn(frozen):
    evt = FreezeEvent(frozen)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


## called when a button is pressed
# @param button the button that was pressed
# @param clicks number of clicks performed
def buttonsFn(button, clicks):
    evt = ButtonEvent(button, clicks)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


## main function
def main():
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, freezeFn, buttonsFn)
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWidget(cast)
    widget.resize(640, 480)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    model = NNUNet()
    model.load_state_dict(torch.load("nnunet_model.pth", map_location=torch.device('cpu')))
    
    # model = torch.load("unext.pt", map_location=torch.device('cpu'))
    # model.load_state_dict(torch.load("unext.pth", map_location=torch.device('cpu')))
    
    model.eval()
    main()
