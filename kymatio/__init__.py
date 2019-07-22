# Make sure that DeprecationWarning within this package always gets printed
### Snippet copied from sklearn.__init__
import warnings
import re
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
### End Snippet

__all__ = [
    'PhaseHarmonics2d',
]

from .version import version as __version__
