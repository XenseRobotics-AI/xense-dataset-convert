"""Convert ROS2 MCAP episodes to LeRobot v3 datasets."""

from .config import ConverterConfig
from .converter import McapToLeRobotConverter

__version__ = "0.1.0"

__all__ = ["ConverterConfig", "McapToLeRobotConverter", "__version__"]
