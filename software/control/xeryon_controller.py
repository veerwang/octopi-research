import os
import sys
import time
import serial

from control.Xeryon import *

class XeryonController:
    def __init__(self, sn):
        super().__init__()
        port = [p.device for p in serial.tools.list_ports.comports() if sn == p.serial_number]
        self.controller = Xeryon(port, 115200)
        self.axisX = self.controller.addAxis(Stage.XLA_1250_3N, "Z")
        self.controller.start()
        self.controller.reset()

        self.position1 = -19
        self.position2 = 19

        self.current_position = 1

    def stopScan(self):
        self.axisX.stopScan()

    def homing(self):
        self.axisX.findIndex()
        self.axisX.stopScan()

    def moveToPosition1(self):
        self.axisX.setDPOS(self.position1)
        self.current_position = 1

    def moveToPosition2(self):
        self.axisX.setDPOS(self.position2)
        self.current_position = 2

    def currentPosition(self):
        return self.current_position

    def setSpeed(self, value):
        self.axisX.setSpeed(value)