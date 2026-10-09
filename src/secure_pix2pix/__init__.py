"""Secure, from-scratch pix2pix training components.

This package is intentionally independent from the existing correction_ml V2
learner.  It never performs inference against production files and all writes
are constrained to an explicitly selected output or staging directory.
"""

from PIL import Image

# Matches correction_ml's ceiling (see correction_ml/__init__.py) so the
# decompression-bomb guard does not depend on which package happens to be
# imported first when this package is used standalone.
Image.MAX_IMAGE_PIXELS = 250_000_000

from .models import PatchDiscriminator, UNetGenerator, initialize_pix2pix_weights

IMAGE_SIZE = 256
LAMBDA_L1 = 100.0
LEARNING_RATE = 0.0002
ADAM_BETAS = (0.5, 0.999)

__all__ = [
    "ADAM_BETAS",
    "IMAGE_SIZE",
    "LAMBDA_L1",
    "LEARNING_RATE",
    "PatchDiscriminator",
    "UNetGenerator",
    "initialize_pix2pix_weights",
]
