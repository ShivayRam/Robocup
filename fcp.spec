# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Run_Player.py'],
    pathex=[],
    binaries=[],
    datas=[('./world/commons/robots', 'world/commons/robots'), ('./behaviors/slot/common', 'behaviors/slot/common'), ('./behaviors/slot/r0', 'behaviors/slot/r0'), ('./behaviors/slot/r1', 'behaviors/slot/r1'), ('./behaviors/slot/r2', 'behaviors/slot/r2'), ('./behaviors/slot/r3', 'behaviors/slot/r3'), ('./behaviors/slot/r4', 'behaviors/slot/r4'), ('./behaviors/custom/Dribble/*.pkl', 'behaviors/custom/Dribble'), ('./behaviors/custom/Walk/*.pkl', 'behaviors/custom/Walk'), ('./behaviors/custom/Fall/*.pkl', 'behaviors/custom/Fall')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fcp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
