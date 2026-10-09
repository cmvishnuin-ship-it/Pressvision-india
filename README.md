# Pressvision-india
Windows service that learns how newspaper photo editors correct images: pairs originals with Photoshop-edited versions, measures 34 correction effects, trains PyTorch models (attention U-Net, whole-picture tone-curve model, pix2pix) and scores them on held-out whole images. Read-only, time-windowed data access; live dashboard; 85 tests.
