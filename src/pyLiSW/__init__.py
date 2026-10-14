"""pyLiSW: linear spin wave theory modeling of inelastic neutron scattering data."""

from importlib.metadata import PackageNotFoundError, version

from .Atoms import Atoms
from .Bonds import Bonds
from .Dimers import Dimers
from .LSWT import LSWT
from .QEspace import QEspace
from .Sample import Sample

try:
    __version__ = version("pyLiSW")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "unknown"

__all__ = ["Atoms", "Bonds", "Dimers", "LSWT", "QEspace", "Sample", "__version__"]
