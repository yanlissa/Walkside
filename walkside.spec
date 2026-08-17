# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_delvewheel_libs_directory,
    collect_submodules,
)


PROJECT_ROOT = Path(SPECPATH).resolve()

# Данные pyogrio: GDAL-конфигурация и прочие служебные файлы.
datas = collect_data_files("pyogrio")

# DLL из каталога pyogrio.libs, включая GDAL.
binaries = []

datas, binaries = collect_delvewheel_libs_directory(
    "pyogrio",
    datas=datas,
    binaries=binaries,
)

hiddenimports = [
    "torch",
    "unicodedata",
    "geopandas",
    "pyogrio",
    "pyproj",
    "shapely",
    *collect_submodules("pyogrio"),
]


a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(
            PROJECT_ROOT
            / "build_support"
            / "runtime_hook_native_libraries.py"
        ),
    ],
    excludes=[
        "fiona",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WalkSide",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    contents_directory="src",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="WalkSide",
)