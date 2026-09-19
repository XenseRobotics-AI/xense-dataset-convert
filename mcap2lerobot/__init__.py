"""Convert ROS2 MCAP episodes to LeRobot v3 datasets."""

from .config import ConverterConfig
from .converter import McapToLeRobotConverter

from xense_dataset_convert._version import resolve_version

__version__ = resolve_version()

__all__ = ["ConverterConfig", "McapToLeRobotConverter", "__version__"]
