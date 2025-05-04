from enum import Enum
import json
import os
from pathlib import Path


class LightSourceType(Enum):
    SquidLED = 0
    SquidLaser = 1
    LDI = 2
    CELESTA = 3
    VersaLase = 4
    SCI = 5


class IntensityControlMode(Enum):
    SquidControllerDAC = 0
    Software = 1


class ShutterControlMode(Enum):
    TTL = 0
    Software = 1


class IlluminationController:
    def __init__(
        self,
        microcontroller,
        intensity_control_mode=IntensityControlMode.SquidControllerDAC,
        shutter_control_mode=ShutterControlMode.TTL,
        light_source_type=None,
        light_source=None,
    ):
        self.microcontroller = microcontroller
        self.intensity_control_mode = intensity_control_mode
        self.shutter_control_mode = shutter_control_mode
        self.light_source_type = light_source_type
        self.light_source = light_source

        # Default channel mappings
        default_mappings = {
            405: 11,
            470: 12,
            488: 12,
            545: 14,
            550: 14,
            555: 14,
            561: 14,
            638: 13,
            640: 13,
            730: 15,
            735: 15,
            750: 15,
        }

        # Try to load mappings from file
        self.channel_mappings_TTL = self._load_channel_mappings(default_mappings)

        self.channel_mappings_software = {}
        self.is_on = {}
        self.intensity_settings = {}
        self.current_channel = None

        if self.light_source_type is not None:
            self._configure_light_source()

    def _load_channel_mappings(self, default_mappings):
        """Load channel mappings from JSON file, fallback to default if file not found."""
        try:
            # Get the parent directory of the current file
            current_dir = Path(__file__).parent.parent
            mapping_file = current_dir / "channel_mappings.json"

            if mapping_file.exists():
                with open(mapping_file, "r") as f:
                    mappings = json.load(f)
                    # Convert string keys to integers
                    return {int(k): v for k, v in mappings["Illumination Code Map"].items()}
            return default_mappings
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return default_mappings

    def _configure_light_source(self):
        self.light_source.initialize()
        self._set_intensity_control_mode(self.intensity_control_mode)
        self._set_shutter_control_mode(self.shutter_control_mode)
        self.channel_mappings_software = self.light_source.channel_mappings
        for ch in self.channel_mappings_software:
            self.intensity_settings[ch] = self.get_intensity(ch)
            self.is_on[ch] = self.light_source.get_shutter_state(self.channel_mappings_software[ch])

    def _set_intensity_control_mode(self, mode):
        self.light_source.set_intensity_control_mode(mode)
        self.intensity_control_mode = mode

    def _set_shutter_control_mode(self, mode):
        self.light_source.set_shutter_control_mode(mode)
        self.shutter_control_mode = mode

    # current not used
    """
    def get_intensity_control_mode(self):
        mode = self.light_source.get_intensity_control_mode()
        if mode is not None:
            self.intensity_control_mode = mode
            return mode

    def get_shutter_control_mode(self):
        mode = self.light_source.get_shutter_control_mode()
        if mode is not None:
            self.shutter_control_mode = mode
            return mode
    """

    def get_intensity(self, channel):
        if self.intensity_control_mode == IntensityControlMode.Software:
            intensity = self.light_source.get_intensity(self.channel_mappings_software[channel])
            self.intensity_settings[channel] = intensity
            return intensity  # 0 - 100

    def turn_on_illumination(self, channel=None):
        if channel is None:
            channel = self.current_channel

        if self.shutter_control_mode == ShutterControlMode.Software:
            self.light_source.set_shutter_state(self.channel_mappings_software[channel], on=True)
        elif self.shutter_control_mode == ShutterControlMode.TTL:
            # self.microcontroller.set_illumination(self.channel_mappings_TTL[channel], self.intensity_settings[channel])
            self.microcontroller.turn_on_illumination()

        self.is_on[channel] = True

    def turn_off_illumination(self, channel=None):
        if channel is None:
            channel = self.current_channel

        if self.shutter_control_mode == ShutterControlMode.Software:
            self.light_source.set_shutter_state(self.channel_mappings_software[channel], on=False)
        elif self.shutter_control_mode == ShutterControlMode.TTL:
            self.microcontroller.turn_off_illumination()

        self.is_on[channel] = False

    def set_intensity(self, channel, intensity):
        if self.intensity_control_mode == IntensityControlMode.Software:
            if intensity != self.intensity_settings[channel]:
                self.light_source.set_intensity(self.channel_mappings_software[channel], intensity)
                self.intensity_settings[channel] = intensity
        else:
            self.microcontroller.set_illumination(self.channel_mappings_TTL[channel], intensity)

    def get_shutter_state(self):
        return self.is_on

    def close(self):
        if self.light_source is not None:
            self.light_source.shut_down()
