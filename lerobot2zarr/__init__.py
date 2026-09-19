"""Generic LeRobot to Zarr converter."""

from xense_dataset_convert._version import resolve_version

__version__ = resolve_version()


from .zarr_converter import LeRobotZarrConverter

__all__ = ["LeRobotZarrConverter", "__version__"]
