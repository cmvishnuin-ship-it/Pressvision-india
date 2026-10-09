"""Learning and analysis tools for paired production image corrections."""

from PIL import Image

# Production photography can legitimately exceed Pillow's conservative default
# warning threshold (~89 MP). Keep a finite safety ceiling rather than disabling
# decompression-bomb protection. Pillow raises an error at twice this value.
Image.MAX_IMAGE_PIXELS = 250_000_000

__version__ = "0.1.0"
