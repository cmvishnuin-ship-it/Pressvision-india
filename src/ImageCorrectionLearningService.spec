# -*- mode: python ; coding: utf-8 -*-

import os

datas = [
    ("config.server.json", "."),
]
# An initial trained V2 checkpoint is bundled when one is present; without it
# the service trains V2 from scratch.
if os.path.isfile("artifacts/best_v2.pt"):
    datas.append(("artifacts/best_v2.pt", "artifacts"))
# Bundled once, on a machine with internet, via the command documented at the
# top of correction_ml/model_pretrained.py -- never downloaded at runtime.
if os.path.isfile("artifacts/pretrained/resnet18_imagenet.pt"):
    datas.append(("artifacts/pretrained/resnet18_imagenet.pt", "artifacts/pretrained"))

a = Analysis(
    ["learning_service.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "analyze",
        "train",
        "evaluate_whole_images",
        "pix2pix_train",
        "secure_pix2pix",
        "secure_pix2pix.checkpoints",
        "secure_pix2pix.data",
        "secure_pix2pix.metrics",
        "secure_pix2pix.models",
        "secure_pix2pix.security",
        "secure_pix2pix.staging",
        "correction_ml.model_pretrained",
        "correction_ml.model_global",
        "torchvision.models.resnet",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorflow",
        "keras",
        "matplotlib",
        "IPython",
        "notebook",
        "jupyter",
        "pytest",
        "pandas",
        "lxml",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "tkinter",
        "zmq",
        "tensorboard",
        "torch.utils.tensorboard",
        "torchaudio",
        "torchtext",
        "torchdata",
        "transformers",
        "timm",
        "fsspec",
        "cryptography",
        "bcrypt",
        "pydantic",
        "anyio",
        "requests",
        "urllib3",
        "jinja2",
        "win32com",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImageCorrectionLearningService",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ImageCorrectionLearningService",
)
