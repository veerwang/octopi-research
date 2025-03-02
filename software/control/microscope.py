import serial

from PyQt5.QtCore import QObject

import control.core.core as core
from control._def import *
import control
from squid.abc import AbstractStage
import squid.stage.cephla
import squid.abc
import squid.logging
import squid.config
import squid.stage.utils

if CAMERA_TYPE == "Toupcam":
    import control.camera_toupcam as camera
if FOCUS_CAMERA_TYPE == "Toupcam":
    try:
        import control.camera_toupcam as camera_fc
    except:
        log.warning("Problem importing Toupcam for focus, defaulting to default camera")
        import control.camera as camera_fc
elif FOCUS_CAMERA_TYPE == "FLIR":
    try:
        import control.camera_flir as camera_fc
    except:
        log.warning("Problem importing FLIR camera for focus, defaulting to default camera")
        import control.camera as camera_fc
else:
    import control.camera as camera_fc

import control.microcontroller as microcontroller
from control.piezo import PiezoStage
import control.serial_peripherals as serial_peripherals
if SUPPORT_LASER_AUTOFOCUS:
    import control.core_displacement_measurement as core_displacement_measurement


class Microscope(QObject):

    def __init__(self, microscope=None, is_simulation=False):
        super().__init__()
        if microscope is None:
            self.initialize_camera(is_simulation=is_simulation)
            self.initialize_microcontroller(is_simulation=is_simulation)
            self.initialize_core_components()
            self.initialize_peripherals()
        else:
            self.camera = microscope.camera
            self.stage = microscope.stage
            self.microcontroller = microscope.microcontroller
            self.configurationManager = microscope.configurationManager
            self.objectiveStore = microscope.objectiveStore
            self.streamHandler = microscope.streamHandler
            self.liveController = microscope.liveController
            self.autofocusController = microscope.autofocusController
            self.slidePositionController = microscope.slidePositionController
            if USE_ZABER_EMISSION_FILTER_WHEEL:
                self.emission_filter_wheel = microscope.emission_filter_wheel
            elif USE_OPTOSPIN_EMISSION_FILTER_WHEEL:
                self.emission_filter_wheel = microscope.emission_filter_wheel

    def initialize_camera(self, is_simulation):
        if is_simulation:
            self.camera = camera.Camera_Simulation(rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE)
        else:
            sn_camera_main = camera.get_sn_by_model(MAIN_CAMERA_MODEL)
            self.camera = camera.Camera(sn=sn_camera_main, rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE)

        self.camera.open()
        self.camera.set_pixel_format(DEFAULT_PIXEL_FORMAT)
        self.camera.set_software_triggered_acquisition()

        if SUPPORT_LASER_AUTOFOCUS:
            if is_simulation:
                self.camera_focus = camera_fc.Camera_Simulation(rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE)
            else:
                sn_camera_focus = camera_fc.get_sn_by_model(FOCUS_CAMERA_MODEL)
                self.camera_focus = camera_fc.Camera(sn=sn_camera_focus, rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE)
            self.camera_focus.open()
            self.camera_focus.set_pixel_format("MONO8")
            self.camera_focus.set_software_triggered_acquisition()

    def initialize_microcontroller(self, is_simulation):
        self.microcontroller = microcontroller.Microcontroller(
            serial_device=microcontroller.get_microcontroller_serial_device(
                version=CONTROLLER_VERSION, sn=CONTROLLER_SN, simulated=is_simulation
            )
        )
        self.stage = squid.stage.cephla.CephlaStage(
            microcontroller=self.microcontroller, stage_config=squid.config.get_stage_config()
        )

        self.home_x_and_y_separately = False

    def initialize_core_components(self):
        if HAS_OBJECTIVE_PIEZO:
            self.piezo = PiezoStage(
                self.microcontroller,
                {
                    "OBJECTIVE_PIEZO_HOME_UM": OBJECTIVE_PIEZO_HOME_UM,
                    "OBJECTIVE_PIEZO_RANGE_UM": OBJECTIVE_PIEZO_RANGE_UM,
                    "OBJECTIVE_PIEZO_CONTROL_VOLTAGE_RANGE": OBJECTIVE_PIEZO_CONTROL_VOLTAGE_RANGE,
                    "OBJECTIVE_PIEZO_FLIP_DIR": OBJECTIVE_PIEZO_FLIP_DIR,
                },
            )
            self.piezo.home()
        else:
            self.piezo = None

        self.objectiveStore = core.ObjectiveStore(parent=self)
        self.channelConfigurationManager = core.ChannelConfigurationManager()
        if SUPPORT_LASER_AUTOFOCUS:
            self.laserAFSettingManager = core.LaserAFSettingManager()
        else:
            self.laserAFSettingManager = None
        self.configurationManager = core.ConfigurationManager(
            self.channelConfigurationManager, self.laserAFSettingManager
        )
        self.streamHandler = core.StreamHandler()
        self.liveController = core.LiveController(self.camera, self.microcontroller, None, self)
        self.autofocusController = core.AutoFocusController(
            self.camera, self.stage, self.liveController, self.microcontroller
        )
        self.slidePositionController = core.SlidePositionController(self.stage, self.liveController)
        
        self.multipointController = core.MultiPointController(
            self.camera,
            self.stage,
            self.piezo,
            self.microcontroller,
            self.liveController,
            self.autofocusController,
            self.objectiveStore,
            self.channelConfigurationManager,
            scanCoordinates=None,
            parent=self,
        )

        if SUPPORT_LASER_AUTOFOCUS:
            self.streamHandler_focus_camera = core.StreamHandler()
            self.liveController_focus_camera = core.LiveController(
                self.camera_focus,
                self.microcontroller,
                self,
                control_illumination=False,
                for_displacement_measurement=True,
            )
            self.imageDisplayWindow_focus = core.ImageDisplayWindow(show_LUT=False, autoLevels=False)
            self.displacementMeasurementController = core_displacement_measurement.DisplacementMeasurementController()
            self.laserAutofocusController = core.LaserAutofocusController(
                self.microcontroller,
                self.camera_focus,
                self.liveController_focus_camera,
                self.stage,
                self.piezo,
                self.objectiveStore,
                self.laserAFSettingManager,
            )

    def initialize_peripherals(self):
        if USE_ZABER_EMISSION_FILTER_WHEEL:
            self.emission_filter_wheel = serial_peripherals.FilterController(
                FILTER_CONTROLLER_SERIAL_NUMBER, 115200, 8, serial.PARITY_NONE, serial.STOPBITS_ONE
            )
            self.emission_filter_wheel.start_homing()
        elif USE_OPTOSPIN_EMISSION_FILTER_WHEEL:
            self.emission_filter_wheel = serial_peripherals.Optospin(SN=FILTER_CONTROLLER_SERIAL_NUMBER)
            self.emission_filter_wheel.set_speed(OPTOSPIN_EMISSION_FILTER_WHEEL_SPEED_HZ)

    def set_channel(self, channel):
        self.liveController.set_channel(channel)

    def acquire_image(self):
        # turn on illumination and send trigger
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_on_illumination()
            self.waitForMicrocontroller()
            self.camera.send_trigger()
        elif self.liveController.trigger_mode == TriggerMode.HARDWARE:
            self.microcontroller.send_hardware_trigger(
                control_illumination=True, illumination_on_time_us=self.camera.exposure_time * 1000
            )

        # read a frame from camera
        image = self.camera.read_frame()
        if image is None:
            print("self.camera.read_frame() returned None")

        # tunr off the illumination if using software trigger
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_off_illumination()

        return image

    def home_xyz(self):
        if HOMING_ENABLED_Z:
            self.stage.home(x=False, y=False, z=True, theta=False)
        if HOMING_ENABLED_X and HOMING_ENABLED_Y:
            self.stage.move_x(20)
            self.stage.home(x=False, y=True, z=False, theta=False)
            self.stage.home(x=True, y=False, z=False, theta=False)
            self.slidePositionController.homing_done = True

    def move_x(self, distance, blocking=True):
        self.stage.move_x(distance, blocking=blocking)

    def move_y(self, distance, blocking=True):
        self.stage.move_y(distance, blocking=blocking)

    def move_x_to(self, position, blocking=True):
        self.stage.move_x_to(position, blocking=blocking)

    def move_y_to(self, position, blocking=True):
        self.stage.move_y_to(position, blocking=blocking)

    def get_x(self):
        return self.stage.get_pos().x_mm

    def get_y(self):
        return self.stage.get_pos().y_mm

    def get_z(self):
        return self.stage.get_pos().z_mm

    def move_z_to(self, z_mm, blocking=True):
        self.stage.move_z_to(z_mm, blocking=blocking)
        clear_backlash = z_mm >= self.stage.get_pos().z_mm
        # clear backlash if moving backward in open loop mode
        if blocking and clear_backlash:
            distance_to_clear_backlash = self.stage.get_config().Z_AXIS.convert_to_real_units(
                max(160, 20 * self.stage.get_config().Z_AXIS.MICROSTEPS_PER_STEP)
            )
            self.stage.move_z(-distance_to_clear_backlash)
            self.stage.move_z(distance_to_clear_backlash)

    def start_live(self):
        self.camera.start_streaming()
        self.liveController.start_live()

    def stop_live(self):
        self.liveController.stop_live()
        self.camera.stop_streaming()

    def waitForMicrocontroller(self, timeout=5.0, error_message=None):
        try:
            self.microcontroller.wait_till_operation_is_completed(timeout)
        except TimeoutError as e:
            self.log.error(error_message or "Microcontroller operation timed out!")
            raise e

    def close(self):
        self.stop_live()
        self.camera.close()
        self.microcontroller.close()
        if USE_ZABER_EMISSION_FILTER_WHEEL or USE_OPTOSPIN_EMISSION_FILTER_WHEEL:
            self.emission_filter_wheel.close()
