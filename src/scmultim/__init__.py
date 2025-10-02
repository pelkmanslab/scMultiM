"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scmultim")
except PackageNotFoundError:
    __version__ = "uninstalled"
