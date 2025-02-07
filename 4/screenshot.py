import pyautogui
import tkinter as tk
from tkinter import messagebox
import io


def take_screenshot():
    try:
        # 进行屏幕截图
        screenshot = pyautogui.screenshot()
        # 将截图保存到内存中
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # 模拟上传文件对象
        class MockUploadedFile:
            def __init__(self, bytes_data, name='screenshot.png', type='image/png'):
                self._bytes_data = bytes_data
                self.name = name
                self.type = type

            def read(self):
                return self._bytes_data

        # 返回模拟的上传文件对象
        return MockUploadedFile(img_byte_arr.read())
    except Exception as e:
        messagebox.showerror('截图失败', f'截图过程中出现错误: {e}')
        return None


def open_screenshot_window():
    root = tk.Tk()
    root.title('截图工具')
    result = None

    def perform_screenshot():
        nonlocal result
        result = take_screenshot()
        root.destroy()

    button = tk.Button(root, text='截图', command=perform_screenshot)
    button.pack(pady=20)
    root.mainloop()
    return result