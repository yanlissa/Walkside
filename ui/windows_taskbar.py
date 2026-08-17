from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


TBPF_NOPROGRESS = 0x0
TBPF_INDETERMINATE = 0x1
TBPF_NORMAL = 0x2
TBPF_ERROR = 0x4
TBPF_PAUSED = 0x8

S_OK = 0
S_FALSE = 1
RPC_E_CHANGED_MODE = -2147417850


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str):
        import uuid

        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class WindowsTaskbarProgress:
    def __init__(self, hwnd: int):
        self.hwnd = wintypes.HWND(hwnd)
        self._ptr = None
        self._set_value = None
        self._set_state = None
        self._co_initialized = False

        if sys.platform != "win32":
            return

        try:
            self._initialize()
        except Exception:
            self.close()

    @property
    def available(self) -> bool:
        return self._ptr is not None

    @staticmethod
    def _failed(hresult: int) -> bool:
        return int(hresult) < 0

    def _initialize(self) -> None:
        ole32 = ctypes.windll.ole32
        co_result = int(ole32.CoInitialize(None))
        if co_result in (S_OK, S_FALSE):
            self._co_initialized = True
        elif co_result != RPC_E_CHANGED_MODE:
            raise OSError(f"CoInitialize failed: 0x{co_result & 0xFFFFFFFF:08X}")

        clsid = GUID.from_string("56FDF344-FD6D-11D0-958A-006097C9A090")
        iid = GUID.from_string("EA1AFB91-9E28-4B86-90E9-9E9F8A5EEA84")
        ptr = ctypes.c_void_p()

        result = int(ole32.CoCreateInstance(
            ctypes.byref(clsid),
            None,
            1,  # CLSCTX_INPROC_SERVER
            ctypes.byref(iid),
            ctypes.byref(ptr),
        ))
        if self._failed(result):
            raise OSError(
                f"CoCreateInstance(ITaskbarList3) failed: 0x{result & 0xFFFFFFFF:08X}"
            )

        vtable = ctypes.cast(
            ptr,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
        ).contents
        call = ctypes.WINFUNCTYPE

        hr_init = call(ctypes.c_long, ctypes.c_void_p)(vtable[3])
        set_progress_value = call(
            ctypes.c_long,
            ctypes.c_void_p,
            wintypes.HWND,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
        )(vtable[9])
        set_progress_state = call(
            ctypes.c_long,
            ctypes.c_void_p,
            wintypes.HWND,
            wintypes.DWORD,
        )(vtable[10])

        result = int(hr_init(ptr))
        if self._failed(result):
            raise OSError(f"ITaskbarList3.HrInit failed: 0x{result & 0xFFFFFFFF:08X}")

        self._ptr = ptr
        self._set_value = set_progress_value
        self._set_state = set_progress_state

    def _state(self, state: int) -> None:
        if self.available:
            self._set_state(self._ptr, self.hwnd, state)

    def set_indeterminate(self) -> None:
        self._state(TBPF_INDETERMINATE)

    def set_progress(self, value: int, maximum: int = 1000) -> None:
        if not self.available:
            return
        maximum = max(1, int(maximum))
        value = min(max(0, int(value)), maximum)
        self._state(TBPF_NORMAL)
        self._set_value(self._ptr, self.hwnd, value, maximum)

    def set_paused(self, value: int, maximum: int = 1000) -> None:
        if not self.available:
            return
        maximum = max(1, int(maximum))
        value = min(max(0, int(value)), maximum)
        self._state(TBPF_PAUSED)
        self._set_value(self._ptr, self.hwnd, value, maximum)

    def set_error(self, value: int = 1000, maximum: int = 1000) -> None:
        if not self.available:
            return
        maximum = max(1, int(maximum))
        value = min(max(0, int(value)), maximum)
        self._state(TBPF_ERROR)
        self._set_value(self._ptr, self.hwnd, value, maximum)

    def clear(self) -> None:
        self._state(TBPF_NOPROGRESS)

    def close(self) -> None:
        if self._ptr is not None:
            try:
                vtable = ctypes.cast(
                    self._ptr,
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
                ).contents
                release = ctypes.WINFUNCTYPE(
                    ctypes.c_ulong,
                    ctypes.c_void_p,
                )(vtable[2])
                release(self._ptr)
            except Exception:
                pass

        self._ptr = None
        self._set_value = None
        self._set_state = None

        if self._co_initialized and sys.platform == "win32":
            try:
                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass
            self._co_initialized = False
