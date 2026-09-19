"""Convert UMI-rig HDF5/MKV recordings into LeRobot v3.0 datasets."""

from .config import ConverterConfig, StreamRule
from .converter import Hdf5ToLeRobotConverter, discover_episodes

from xense_dataset_convert._version import resolve_version

__version__ = resolve_version()

__all__ = [
    "ConverterConfig",
    "StreamRule",
    "Hdf5ToLeRobotConverter",
    "discover_episodes",
]
