"""Generic LeRobot to Zarr converter."""

from importlib.metadata import PackageNotFoundError, version

try:
    # The distribution is xense-dataset-convert; this package is one of those it
    # ships, so the version is looked up by that name rather than this module's.
    # A hardcoded string here is worse than none: it is written into every
    # dataset this package converts, and it went three releases without being
    # touched, so it said 0.1.0 about work done at 0.3.13.
    __version__ = version("xense-dataset-convert")
except PackageNotFoundError:  # used from a checkout, not installed
    __version__ = "unknown"


from .zarr_converter import LeRobotZarrConverter

__all__ = ["LeRobotZarrConverter", "__version__"]
