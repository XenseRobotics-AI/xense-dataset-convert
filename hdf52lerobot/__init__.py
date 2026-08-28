"""Convert UMI-rig HDF5/MKV recordings into LeRobot v3.0 datasets."""

from .config import ConverterConfig, StreamRule
from .converter import Hdf5ToLeRobotConverter, discover_episodes

__version__ = "0.1.0"

__all__ = [
    "ConverterConfig",
    "StreamRule",
    "Hdf5ToLeRobotConverter",
    "discover_episodes",
]
