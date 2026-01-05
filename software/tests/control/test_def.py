"""Tests for control._def module."""

import pytest
from control._def import ZMotorConfig, HardwareTriggerMode


class TestZMotorConfig:
    """Tests for ZMotorConfig enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert ZMotorConfig.STEPPER.value == "STEPPER"
        assert ZMotorConfig.STEPPER_PIEZO.value == "STEPPER + PIEZO"
        assert ZMotorConfig.PIEZO.value == "PIEZO"

    def test_convert_to_enum_from_string(self):
        """Test conversion from string values."""
        assert ZMotorConfig.convert_to_enum("STEPPER") == ZMotorConfig.STEPPER
        assert ZMotorConfig.convert_to_enum("STEPPER + PIEZO") == ZMotorConfig.STEPPER_PIEZO
        assert ZMotorConfig.convert_to_enum("PIEZO") == ZMotorConfig.PIEZO

    def test_convert_to_enum_from_enum(self):
        """Test that convert_to_enum returns enum unchanged."""
        assert ZMotorConfig.convert_to_enum(ZMotorConfig.STEPPER) == ZMotorConfig.STEPPER
        assert ZMotorConfig.convert_to_enum(ZMotorConfig.PIEZO) == ZMotorConfig.PIEZO

    def test_convert_to_enum_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Z motor config"):
            ZMotorConfig.convert_to_enum("INVALID")
        with pytest.raises(ValueError, match="Invalid Z motor config"):
            ZMotorConfig.convert_to_enum("stepper")  # Case sensitive

    def test_has_piezo(self):
        """Test has_piezo() method."""
        assert ZMotorConfig.STEPPER.has_piezo() is False
        assert ZMotorConfig.STEPPER_PIEZO.has_piezo() is True
        assert ZMotorConfig.PIEZO.has_piezo() is True

    def test_is_piezo_only(self):
        """Test is_piezo_only() method."""
        assert ZMotorConfig.STEPPER.is_piezo_only() is False
        assert ZMotorConfig.STEPPER_PIEZO.is_piezo_only() is False
        assert ZMotorConfig.PIEZO.is_piezo_only() is True


class TestHardwareTriggerMode:
    """Tests for HardwareTriggerMode class."""

    def test_enum_values(self):
        """Test that enum has expected values matching firmware."""
        assert HardwareTriggerMode.EDGE == 0
        assert HardwareTriggerMode.LEVEL == 1

    def test_values_match_firmware_protocol(self):
        """Test that values can be passed directly to microcontroller."""
        # These values are sent directly to firmware via set_trigger_mode()
        # EDGE (0) = fixed pulse width (TRIGGER_PULSE_LENGTH_us)
        # LEVEL (1) = variable pulse width (illumination_on_time)
        assert isinstance(HardwareTriggerMode.EDGE, int)
        assert isinstance(HardwareTriggerMode.LEVEL, int)
        assert HardwareTriggerMode.EDGE in (0, 1)
        assert HardwareTriggerMode.LEVEL in (0, 1)
        assert HardwareTriggerMode.EDGE != HardwareTriggerMode.LEVEL
