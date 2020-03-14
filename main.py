
import sys, time
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import copy
import msvcrt
import numpy
import cv2
sys.path.append("./MvImport")
from MvCameraControl_class import *
from infer_video import inference
from PIL import Image

from ui_main import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.VideoTimer = Video()  # videotimer是一个对象
        self.VideoTimer.changePixmap.connect(self.setImage)  # changePixmap信号，槽函数是setImage
        self.VideoTimer.detectPixmap.connect(self.setDetect)  # detectPixmap是一个信号，槽函数是SetDetect

    def setImage(self, frame):
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                   QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
        p = convertToQtFormat.scaled(900, 650)  # Qt.KeepAspectRatio 保持图片的尺寸
        self.label.setPixmap(QPixmap.fromImage(p).scaled(self.label.width(), self.label.height()))

    def setDetect(self, image):
        self.img = image
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                   QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
        p = convertToQtFormat.scaled(900, 650)  # Qt.KeepAspectRatio 保持图片的尺寸
        self.label_2.setPixmap(QPixmap.fromImage(p).scaled(self.label_2.width(), self.label_2.height()))

    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.VideoTimer.start()  # 展示线程

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.VideoTimer.setDetect()  # 检测线程

    # @pyqtSlot()
    # def on_pushButtonSave_clicked(self):
    #     """
    #     Slot documentation goes here.
    #     """
    #     # TODO: not implemented yet
    #     # self.VideoTimer.setSave()
    #
    #     name = int(time.time())
    #     cv2.imwrite('./result/' + str(name) + '.jpg', self.img)


class Video(QThread):
    changePixmap = pyqtSignal(numpy.ndarray)  # 一般来说最简单的信号有clicked pressed released  现在相当于是自定义 pyqtSignal是高级自定义信号
    detectPixmap = pyqtSignal(numpy.ndarray)  #

    def __init__(self):
        QThread.__init__(self)
        self.detect = 0
        self.saveflag = 0
        self.de =inference()

    def run(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()  # 相机列表
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE  # 相机类型

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_UWSB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)

        nConnectionNum = 0  # input("please input the number of the device to connect:")

        if int(nConnectionNum) >= deviceList.nDeviceNum:
            print("intput error!")
            sys.exit()

        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        stDeviceList = MV_FRAME_OUT_INFO_EX()
        memset(byref(stDeviceList), 0, sizeof(stDeviceList))
        data_buf = (c_ubyte * nPayloadSize)()
        count = 0
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stDeviceList, 1000)
            if ret == 0:
                # print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (stDeviceList.nWidth, stDeviceList.nHeight, stDeviceList.nFrameNum))
                count += 1
                nRGBSize = stDeviceList.nWidth * stDeviceList.nHeight * 3
                stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                stConvertParam.nWidth = stDeviceList.nWidth
                stConvertParam.nHeight = stDeviceList.nHeight
                stConvertParam.pSrcData = data_buf
                stConvertParam.nSrcDataLen = stDeviceList.nFrameLen
                stConvertParam.enSrcPixelType = stDeviceList.enPixelType
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
                stConvertParam.nDstBufferSize = nRGBSize

                ret = cam.MV_CC_ConvertPixelType(stConvertParam)
                if ret != 0:
                    # print ("convert pixel fail! ret[0x%x]" % ret)
                    del data_buf
                    sys.exit()

                # file_path = "AfterConvert_RGB.raw"
                # file_open = open(file_path.encode('ascii'), 'wb+')
                try:
                    # cap = cv2.VideoCapture("0")

                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    b = numpy.array(img_buff)
                    c = b.reshape(2048, 2448, 3)
                    self.changePixmap.emit(c)  # 理解发射，我触发changepixmap的目的是setimage ，但是我没有参数，所以需要信号先带去收集一个参数
                    if self.detect == 1:
                        # img = numpy.zeros(c.shape)
                        img = c.copy()
                        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        m = self.de.infer(image)
                        imgResult = cv2.cvtColor(numpy.asarray(m), cv2.COLOR_RGB2BGR)
                        print("return results")
                        self.detectPixmap.emit(imgResult)

                #                    name = './result/'+str(count)+'.jpg'
                #                    count+=1
                #                    cv2.imwrite(name,c)
                # file_open.write(img_buff)
                except:
                    raise Exception("save file executed failed:%s" % e.message)

                    # file_open.close()
            else:
                print("get one frame fail, ret[0x%x]" % ret)

            # print ("convert pixeltype succeed!")

            # print ("press a key to continue.")
            # msvcrt.getch()

        # ch:停止取流 | en:Stop grab image
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            del data_buf
            sys.exit()

        # ch:关闭设备 | Close device
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close deivce fail! ret[0x%x]" % ret)
            del data_buf
            sys.exit()

        # ch:销毁句柄 | Destroy handle
        ret = cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            del data_buf
            sys.exit()

        del data_buf

    def setDetect(self):
        self.detect = 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

