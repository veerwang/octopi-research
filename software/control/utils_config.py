from pydantic_xml import BaseXmlModel, element, attr
from typing import List, Optional
from pathlib import Path
import control.utils as utils


class ChannelMode(BaseXmlModel, tag='mode'):
    """Channel configuration model"""
    id: str = attr(name='ID')
    name: str = attr(name='Name')
    exposure_time: float = attr(name='ExposureTime')
    analog_gain: float = attr(name='AnalogGain')
    illumination_source: int = attr(name='IlluminationSource')
    illumination_intensity: float = attr(name='IlluminationIntensity')
    camera_sn: Optional[str] = attr(name='CameraSN', default=None)
    z_offset: float = attr(name='ZOffset')
    emission_filter_position: int = attr(name='EmissionFilterPosition', default=1)
    selected: bool = attr(name='Selected', default=False)
    color: Optional[str] = None  # Not stored in XML but computed from name

    def __init__(self, **data):
        super().__init__(**data)
        self.color = utils.get_channel_color(self.name)

class ChannelConfig(BaseXmlModel, tag='modes'):
    """Root configuration file model"""
    modes: List[ChannelMode] = element(tag='mode')

def get_attr_name(attr_name: str) -> str:
    """Get the attribute name for a given configuration attribute"""
    attr_map = {
        'ID': 'id',
        'Name': 'name',
        'ExposureTime': 'exposure_time',
        'AnalogGain': 'analog_gain',
        'IlluminationSource': 'illumination_source',
        'IlluminationIntensity': 'illumination_intensity',
        'CameraSN': 'camera_sn',
        'ZOffset': 'z_offset',
        'EmissionFilterPosition': 'emission_filter_position',
        'Selected': 'selected',
        'Color': 'color'
    }
    return attr_map[attr_name]

def generate_default_configuration(filename: str) -> None:
    """Generate default configuration using Pydantic models"""
    default_modes = [
        ChannelMode(
            id="1",
            name="BF LED matrix full",
            exposure_time=12,
            analog_gain=0,
            illumination_source=0,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="4",
            name="DF LED matrix",
            exposure_time=22,
            analog_gain=0,
            illumination_source=3,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="5",
            name="Fluorescence 405 nm Ex",
            exposure_time=100,
            analog_gain=10,
            illumination_source=11,
            illumination_intensity=100,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="6",
            name="Fluorescence 488 nm Ex",
            exposure_time=100,
            analog_gain=10,
            illumination_source=12,
            illumination_intensity=100,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="7",
            name="Fluorescence 638 nm Ex",
            exposure_time=100,
            analog_gain=10,
            illumination_source=13,
            illumination_intensity=100,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="8",
            name="Fluorescence 561 nm Ex",
            exposure_time=100,
            analog_gain=10,
            illumination_source=14,
            illumination_intensity=100,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="12",
            name="Fluorescence 730 nm Ex",
            exposure_time=50,
            analog_gain=10,
            illumination_source=15,
            illumination_intensity=100,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="9",
            name="BF LED matrix low NA",
            exposure_time=20,
            analog_gain=0,
            illumination_source=4,
            illumination_intensity=20,
            camera_sn="",
            z_offset=0.0
        ),
        # Commented out modes for reference
        # ChannelMode(
        #     id="10",
        #     name="BF LED matrix left dot",
        #     exposure_time=20,
        #     analog_gain=0,
        #     illumination_source=5,
        #     illumination_intensity=20,
        #     camera_sn="",
        #     z_offset=0.0
        # ),
        # ChannelMode(
        #     id="11",
        #     name="BF LED matrix right dot",
        #     exposure_time=20,
        #     analog_gain=0,
        #     illumination_source=6,
        #     illumination_intensity=20,
        #     camera_sn="",
        #     z_offset=0.0
        # ),
        ChannelMode(
            id="2",
            name="BF LED matrix left half",
            exposure_time=16,
            analog_gain=0,
            illumination_source=1,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="3",
            name="BF LED matrix right half",
            exposure_time=16,
            analog_gain=0,
            illumination_source=2,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="12",
            name="BF LED matrix top half",
            exposure_time=20,
            analog_gain=0,
            illumination_source=7,
            illumination_intensity=20,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="13",
            name="BF LED matrix bottom half",
            exposure_time=20,
            analog_gain=0,
            illumination_source=8,
            illumination_intensity=20,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="14",
            name="BF LED matrix full_R",
            exposure_time=12,
            analog_gain=0,
            illumination_source=0,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="15",
            name="BF LED matrix full_G",
            exposure_time=12,
            analog_gain=0,
            illumination_source=0,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="16",
            name="BF LED matrix full_B",
            exposure_time=12,
            analog_gain=0,
            illumination_source=0,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="21",
            name="BF LED matrix full_RGB",
            exposure_time=12,
            analog_gain=0,
            illumination_source=0,
            illumination_intensity=5,
            camera_sn="",
            z_offset=0.0
        ),
        ChannelMode(
            id="20",
            name="USB Spectrometer",
            exposure_time=20,
            analog_gain=0,
            illumination_source=6,
            illumination_intensity=0,
            camera_sn="",
            z_offset=0.0
        )
    ]

    config = ChannelConfig(modes=default_modes)
    xml_str = config.to_xml(pretty_print=True, encoding='utf-8')
    
    # Write to file
    path = Path(filename)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_bytes(xml_str)