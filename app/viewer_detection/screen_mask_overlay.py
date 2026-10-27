import ctypes
from ctypes import wintypes


# ============================================================
# WINDOWS CONSTANTS
# ============================================================

WS_POPUP = 0x80000000

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

HWND_TOPMOST = -1

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

ULW_ALPHA = 0x00000002

AC_SRC_OVER = 0
AC_SRC_ALPHA = 1

WDA_EXCLUDEFROMCAPTURE = 0x00000011

BI_RGB = 0
DIB_RGB_COLORS = 0


# ============================================================
# WINDOWS STRUCTURES
# ============================================================

class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class SIZE(ctypes.Structure):
    _fields_ = [
        ("cx", wintypes.LONG),
        ("cy", wintypes.LONG),
    ]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", wintypes.BYTE),
        ("BlendFlags", wintypes.BYTE),
        ("SourceConstantAlpha", wintypes.BYTE),
        ("AlphaFormat", wintypes.BYTE),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 1),
    ]


# ============================================================
# WINDOWS DLLS
# ============================================================

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


# ============================================================
# WINDOWS API FUNCTION SIGNATURES
# ============================================================

user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]

user32.CreateWindowExW.restype = wintypes.HWND


user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]

user32.SetWindowPos.restype = wintypes.BOOL


user32.GetWindowLongW.argtypes = [
    wintypes.HWND,
    ctypes.c_int,
]

user32.GetWindowLongW.restype = wintypes.LONG


user32.SetWindowLongW.argtypes = [
    wintypes.HWND,
    ctypes.c_int,
    wintypes.LONG,
]

user32.SetWindowLongW.restype = wintypes.LONG


user32.SetWindowDisplayAffinity.argtypes = [
    wintypes.HWND,
    wintypes.DWORD,
]

user32.SetWindowDisplayAffinity.restype = wintypes.BOOL


user32.DestroyWindow.argtypes = [
    wintypes.HWND,
]

user32.DestroyWindow.restype = wintypes.BOOL


user32.GetDC.argtypes = [
    wintypes.HWND,
]

user32.GetDC.restype = wintypes.HDC


user32.ReleaseDC.argtypes = [
    wintypes.HWND,
    wintypes.HDC,
]

user32.ReleaseDC.restype = ctypes.c_int


user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND,
    wintypes.HDC,
    ctypes.POINTER(POINT),
    ctypes.POINTER(SIZE),
    wintypes.HDC,
    ctypes.POINTER(POINT),
    wintypes.COLORREF,
    ctypes.POINTER(BLENDFUNCTION),
    wintypes.DWORD,
]

user32.UpdateLayeredWindow.restype = wintypes.BOOL


gdi32.CreateCompatibleDC.argtypes = [
    wintypes.HDC,
]

gdi32.CreateCompatibleDC.restype = wintypes.HDC


gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.POINTER(BITMAPINFO),
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD,
]

gdi32.CreateDIBSection.restype = wintypes.HBITMAP


gdi32.SelectObject.argtypes = [
    wintypes.HDC,
    wintypes.HGDIOBJ,
]

gdi32.SelectObject.restype = wintypes.HGDIOBJ


gdi32.DeleteObject.argtypes = [
    wintypes.HGDIOBJ,
]

gdi32.DeleteObject.restype = wintypes.BOOL


gdi32.DeleteDC.argtypes = [
    wintypes.HDC,
]

gdi32.DeleteDC.restype = wintypes.BOOL


# ============================================================
# DPI AWARENESS
# ============================================================

try:
    user32.SetProcessDPIAware()
except Exception:
    pass


# ============================================================
# SCREEN MASK OVERLAY
# ============================================================

class ScreenMaskOverlay:

    def __init__(
        self,
        screen_width,
        screen_height,
        screen_x=0,
        screen_y=0
    ):

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        self.screen_x = int(screen_x)
        self.screen_y = int(screen_y)

        # ----------------------------------------------------
        # Create actual Windows window
        # ----------------------------------------------------

        self.hwnd = self._create_window()

        if not self.hwnd:

            error_code = ctypes.get_last_error()

            raise RuntimeError(
                "Failed to create SafeSight overlay window. "
                f"Windows error: {error_code}"
            )

        # ----------------------------------------------------
        # Configure extended styles
        # ----------------------------------------------------

        ex_style = user32.GetWindowLongW(
            self.hwnd,
            -20
        )

        ex_style |= (
            WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOOLWINDOW
            | WS_EX_NOACTIVATE
        )

        user32.SetWindowLongW(
            self.hwnd,
            -20,
            ex_style
        )

        # ----------------------------------------------------
        # Prevent overlay from appearing in screen capture
        # ----------------------------------------------------

        try:
            user32.SetWindowDisplayAffinity(
                self.hwnd,
                WDA_EXCLUDEFROMCAPTURE
            )
        except Exception:
            pass

        # ----------------------------------------------------
        # Start completely transparent
        # ----------------------------------------------------

        self.clear()

    # ========================================================
    # CREATE WINDOWS WINDOW
    # ========================================================

    def _create_window(self):

        hwnd = user32.CreateWindowExW(

            WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOOLWINDOW
            | WS_EX_NOACTIVATE,

            "STATIC",

            "SafeSightMaskOverlay",

            WS_POPUP,

            self.screen_x,
            self.screen_y,

            self.screen_width,
            self.screen_height,

            None,
            None,
            None,
            None
        )

        return hwnd

    # ========================================================
    # CREATE TRANSPARENT BUFFER
    # ========================================================

    def _create_buffer(self):

        total_bytes = (
            self.screen_width
            * self.screen_height
            * 4
        )

        return bytearray(total_bytes)

    # ========================================================
    # DRAW BLACK RECTANGLE
    # ========================================================

    def _draw_rectangle(
        self,
        buffer,
        x1,
        y1,
        x2,
        y2,
        alpha=255
    ):

        width = self.screen_width
        height = self.screen_height

        # ----------------------------------------------------
        # Clamp coordinates
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                width,
                int(x1)
            )
        )

        y1 = max(
            0,
            min(
                height,
                int(y1)
            )
        )

        x2 = max(
            0,
            min(
                width,
                int(x2)
            )
        )

        y2 = max(
            0,
            min(
                height,
                int(y2)
            )
        )

        if x2 <= x1 or y2 <= y1:
            return

        row_size = width * 4

        # ----------------------------------------------------
        # Completely opaque black pixels
        # ----------------------------------------------------

        for y in range(y1, y2):

            start = (
                y * row_size
                + x1 * 4
            )

            end = (
                y * row_size
                + x2 * 4
            )

            for index in range(
                start,
                end,
                4
            ):

                # BGRA

                buffer[index] = 0
                buffer[index + 1] = 0
                buffer[index + 2] = 0

                # FULLY OPAQUE
                buffer[index + 3] = 255

    # ========================================================
    # UPDATE LAYERED WINDOW
    # ========================================================

    def _update_window(
        self,
        buffer
    ):

        width = self.screen_width
        height = self.screen_height

        # ----------------------------------------------------
        # Get desktop DC
        # ----------------------------------------------------

        screen_dc = user32.GetDC(None)

        if not screen_dc:

            raise RuntimeError(
                "GetDC failed."
            )

        # ----------------------------------------------------
        # Create memory DC
        # ----------------------------------------------------

        memory_dc = gdi32.CreateCompatibleDC(
            screen_dc
        )

        if not memory_dc:

            user32.ReleaseDC(
                None,
                screen_dc
            )

            raise RuntimeError(
                "CreateCompatibleDC failed."
            )

        # ----------------------------------------------------
        # Bitmap information
        # ----------------------------------------------------

        bitmap_info = BITMAPINFO()

        bitmap_info.bmiHeader.biSize = (
            ctypes.sizeof(
                BITMAPINFOHEADER
            )
        )

        bitmap_info.bmiHeader.biWidth = width

        # Top-down bitmap
        bitmap_info.bmiHeader.biHeight = -height

        bitmap_info.bmiHeader.biPlanes = 1

        bitmap_info.bmiHeader.biBitCount = 32

        bitmap_info.bmiHeader.biCompression = BI_RGB

        bits = ctypes.c_void_p()

        # ----------------------------------------------------
        # Create DIB section
        # ----------------------------------------------------

        bitmap = gdi32.CreateDIBSection(

            memory_dc,

            ctypes.byref(
                bitmap_info
            ),

            DIB_RGB_COLORS,

            ctypes.byref(
                bits
            ),

            None,

            0
        )

        if not bitmap:

            gdi32.DeleteDC(
                memory_dc
            )

            user32.ReleaseDC(
                None,
                screen_dc
            )

            raise RuntimeError(
                "CreateDIBSection failed."
            )

        # ----------------------------------------------------
        # Select bitmap
        # ----------------------------------------------------

        old_bitmap = gdi32.SelectObject(
            memory_dc,
            bitmap
        )

        # ----------------------------------------------------
        # Copy pixel buffer
        # ----------------------------------------------------

        ctypes.memmove(
            bits,
            bytes(buffer),
            len(buffer)
        )

        # ----------------------------------------------------
        # Layered window parameters
        # ----------------------------------------------------

        destination = POINT(
            self.screen_x,
            self.screen_y
        )

        size = SIZE(
            width,
            height
        )

        source = POINT(
            0,
            0
        )

        blend = BLENDFUNCTION(
            AC_SRC_OVER,
            0,
            255,
            AC_SRC_ALPHA
        )

        # ----------------------------------------------------
        # Update layered window
        # ----------------------------------------------------

        result = user32.UpdateLayeredWindow(

            self.hwnd,

            screen_dc,

            ctypes.byref(
                destination
            ),

            ctypes.byref(
                size
            ),

            memory_dc,

            ctypes.byref(
                source
            ),

            0,

            ctypes.byref(
                blend
            ),

            ULW_ALPHA
        )

        error_code = ctypes.get_last_error()

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        gdi32.SelectObject(
            memory_dc,
            old_bitmap
        )

        gdi32.DeleteObject(
            bitmap
        )

        gdi32.DeleteDC(
            memory_dc
        )

        user32.ReleaseDC(
            None,
            screen_dc
        )

        if not result:

            raise RuntimeError(
                "UpdateLayeredWindow failed. "
                f"Windows error: {error_code}"
            )

    # ========================================================
    # SHOW OVERLAY
    # ========================================================

    def show(self):

        result = user32.SetWindowPos(

            self.hwnd,

            HWND_TOPMOST,

            self.screen_x,
            self.screen_y,

            self.screen_width,
            self.screen_height,

            SWP_NOACTIVATE
            | SWP_SHOWWINDOW
        )

        if not result:

            error_code = ctypes.get_last_error()

            raise RuntimeError(
                "SetWindowPos failed. "
                f"Windows error: {error_code}"
            )

    # ========================================================
    # CLEAR ALL MASKS
    # ========================================================

    def clear(self):

        buffer = self._create_buffer()

        self._update_window(
            buffer
        )

    # ========================================================
    # TEST RECTANGLE
    # ========================================================

    def test_rectangle(
        self,
        x,
        y,
        width,
        height
    ):

        buffer = self._create_buffer()

        self._draw_rectangle(

            buffer,

            x,
            y,

            x + width,
            y + height,

            # FULLY OPAQUE
            alpha=255
        )

        self._update_window(
            buffer
        )

    # ========================================================
    # LIVE SENSITIVE INFORMATION MASKING
    # ========================================================

    def update_masks(
        self,
        detections
    ):

        buffer = self._create_buffer()

        for detection in detections:

            x, y, width, height = (
                detection["bbox"]
            )

            # ------------------------------------------------
            # Padding around sensitive information
            # ------------------------------------------------

            padding = 5

            # ------------------------------------------------
            # FULLY OPAQUE BLACK MASK
            # ------------------------------------------------

            self._draw_rectangle(

                buffer,

                x - padding,
                y - padding,

                x + width + padding,
                y + height + padding,

                # IMPORTANT:
                # 255 = completely opaque
                # 230 = partially transparent
                alpha=255
            )

        self._update_window(
            buffer
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            self.clear()

        except Exception:

            pass

        try:

            user32.DestroyWindow(
                self.hwnd
            )

        except Exception:

            pass