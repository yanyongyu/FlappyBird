# -*- coding: utf-8 -*-
"""
This is the share module of the game.
Author: yanyongyu
"""

__author__ = "yanyongyu"
__all__ = ["copy", "save"]

import sys
import time

from PIL import Image

# 复制到剪切板
if "win" in sys.platform:
    import win32con
    import win32clipboard
    from io import BytesIO

    def copy(image):
        """
        Only work on Windows.
        Using win32.
        """
        img = Image.fromarray(image)
        img = img.transpose(Image.ROTATE_270)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        output = BytesIO()
        img.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_DIB, data)
        win32clipboard.CloseClipboard()


# 保存图片
def save(image):
    """Save the image to local path."""
    img = Image.fromarray(image)
    img = img.transpose(Image.ROTATE_270)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save("%s.jpg" % round(time.time()))
