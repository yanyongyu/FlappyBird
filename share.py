# -*- coding: utf-8 -*-
"""
This is the share module of the game.
Author: yanyongyu
"""

__author__ = "yanyongyu"
__all__ = ["copy", "save", "send_email"]

import re
import sys
import time
import logging
import threading
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox

import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr

from PIL import Image

logging.basicConfig(level=logging.INFO)

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
        logging.info("Copied successfully")
        root = Tk()
        root.withdraw()
        messagebox.showinfo("Flappy Bird", "复制成功！")
        root.destroy()


# 保存图片
def save(image):
    """Save the image to local path."""
    img = Image.fromarray(image).convert("RGB")
    img = img.transpose(Image.ROTATE_270)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save("%s.jpg" % round(time.time()))
    logging.info("Saved successfully")
    root = Tk()
    root.withdraw()
    messagebox.showinfo("Flappy Bird", "保存成功！")
    root.destroy()


def send_email(image_data, score):
    start_thread(Email, image_data, score)


def start_thread(target, *args, **kw):
    t = threading.Thread(target=target, args=args, kwargs=kw)
    t.start()


class AutoShowScrollbar(Scrollbar):
    # 如果不需要滚动条则会自动隐藏
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("pack", "forget", self)
        else:
            self.pack(fill=Y, side=RIGHT, expand=False)
        Scrollbar.set(self, lo, hi)


class Email():
    """Make a email share to others."""
    html_text = (
        '<html><body><h1>%s</h1>' +
        '<h2>游戏源码地址：' +
        '<a href="https://github.com/yanyongyu/FlappyBird">GitHub</a></h2>' +
        '<a><img src="cid:flappy" alt="flappy"></a>' +
        '<p>Coding Email...</p>' +
        '<p>send by <a href="http://www.python.org">Python</a> app...</p>' +
        '</body></html>')

    smtp_servers = {'126': 'smtp.126.com', 'qq': 'smtp.qq.com',
                    'sina': 'smtp.sina.com.cn', 'aliyun': 'smtp.aliyun.com',
                    '163': 'smtp.163.com', 'yahoo': 'smtp.mail.yahoo.com',
                    'foxmail': 'SMTP.foxmail.com', 'sohu': 'smtp.sohu.com',
                    '139': 'SMTP.139.com', 'china': 'smtp.china.com'}

    def __init__(self, image, score):
        self.score = score
        img = Image.fromarray(image)
        img = img.transpose(Image.ROTATE_270)
        self.img = img.transpose(Image.FLIP_LEFT_RIGHT)
        self.email_check = re.compile(
                r"^[\w]+\.?[\w]+@([\w]+)((\.\w{2,3}){1,3})$")
        logging.info("Show email window")
        self.show()

    def show(self):
        self.root = Tk()
        self.root.title("email share")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - 400) / 2 - 25
        y = (sh - 250) / 2 - 25
        self.root.geometry('%dx%d+%d+%d' % (400, 250, x, y))
        self.root.resizable(False, False)
        self.root.iconbitmap("images/flappy.ico")

        # 邮件信息框架
        frame1 = Frame(self.root)
        frame1.pack(fill=BOTH)
        # 调整entry列权重，使其拉伸
        frame1.columnconfigure(1, weight=1)

        # 发件人邮箱输入行
        label1 = Label(frame1, text="发件人邮箱：")
        label1.grid(row=0, column=0, padx=2, pady=4, sticky=W+N+S)
        self.send_email = StringVar()
        entry1 = Entry(frame1, textvariable=self.send_email)
        entry1.grid(row=0, column=1, padx=2, pady=4, sticky=E+N+S+W)

        # 发件人邮箱密码输入行
        label2 = Label(frame1, text="发件人密码：")
        label2.grid(row=1, column=0, padx=2, pady=4, sticky=W+N+S)
        self.send_pw = StringVar()
        self.entry2 = Entry(frame1, textvariable=self.send_pw, show='*')
        self.entry2.grid(row=1, column=1, padx=2, pady=4, sticky=E+N+S+W)
        self.v = IntVar()
        cb = Checkbutton(frame1, text='显示密码', variable=self.v)
        cb.grid(row=1, column=2, padx=2, pady=4, sticky=E+N+S)
        cb.bind('<ButtonRelease-1>', self.check_show)

        # 收件人邮箱输入行
        label3 = Label(frame1, text="收件人邮箱：")
        label3.grid(row=2, column=0, padx=2, pady=4, sticky=W+N+S)
        self.target_email = StringVar()
        entry3 = Entry(frame1, textvariable=self.target_email)
        entry3.grid(row=2, column=1, padx=2, pady=4, sticky=E+N+S+W)

        # 邮件内容输入框架
        frame2 = Frame(self.root)
        frame2.pack(fill=BOTH, expand=True)

        # 邮件内容输入
        self.text = Text(frame2, width=40, height=5,
                         borderwidth=3, font=('微软雅黑', 12))
        self.text.pack(padx=2, pady=5, side=LEFT, fill=BOTH, expand=True)
        self.text.insert(
                1.0,
                "我在玩Flappy Bird小游戏，取得了%s分的好成绩哟" % self.score)
        vbar_y = AutoShowScrollbar(frame2, orient=VERTICAL)
        vbar_y.pack(fill=Y, side=RIGHT, expand=False)
        vbar_y.config(command=self.text.yview)
        self.text.configure(yscrollcommand=vbar_y.set)

        # 界面鼠标滚动
        def _scroll_text(event):
            self.text.yview_scroll(int(-event.delta / 120), 'units')
        self.text.bind('<MouseWheel>', _scroll_text)

        # 点击发送按钮
        button = Button(self.root, text="点击发送",
                        command=lambda: start_thread(self.send))
        button.pack(pady=4, side=BOTTOM)

        self.root.mainloop()

    def check_show(self, event):
        show = self.v.get()
        if show == 0:
            self.entry2['show'] = ''
        else:
            self.entry2['show'] = '*'

    def _format_addr(self, s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    def send(self):
        logging.info("Start send email")
        top = Toplevel(self.root)
        top.geometry('100x75')
        top.resizable(False, False)
        lb = Label(top, text="正在发送...")
        lb.pack(fill=BOTH)

        from_addr = self.send_email.get()
        to_addr = self.target_email.get()
        logging.info("From email address: %s" % from_addr)
        logging.info("To email address: %s" % to_addr)
        if (not self.email_check.match(from_addr)
                or not self.email_check.match(to_addr)):
            messagebox.showerror("Flappy Bird", "请检查邮箱格式！")
            return
        group = self.email_check.match(from_addr).groups()
        password = self.send_pw.get()
        try:
            smtp_server = Email.smtp_servers[group[0]]
            logging.info("SMTP server: %s" % smtp_server)
        except KeyError:
            messagebox.showerror("Flappy Bird", "该邮箱暂不支持，请联系作者！")
            return

        msg = MIMEMultipart()
        msg.attach(MIMEText(Email.html_text % self.text.get(1.0, END),
                            'html', 'utf-8'))
        msg['From'] = self._format_addr('Python爱好者 <%s>' % from_addr)
        msg['To'] = self._format_addr('管理员 <%s>' % to_addr)
        msg['Subject'] = Header('Flappy Bird', 'utf-8').encode()

        # 设置附件的MIME和文件名，这里是jpg类型:
        logging.info("Write jpg picture into email")
        output = BytesIO()
        self.img.convert("RGB").save(output, "JPEG")
        mime = MIMEImage(output.getvalue(), _subtype="JPEG")
        output.close()
        mime.add_header('Content-ID', 'flappy')
        mime.add_header('Content-Disposition', 'attachment',
                        filename='%s.jpg' % round(time.time()))
        # 添加到MIMEMultipart:
        msg.attach(mime)

        try:
            logging.info("Send email")
            server = smtplib.SMTP(smtp_server, 25)
#            server.set_debuglevel(1)
            server.login(from_addr, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
            server.quit()
            logging.info("Send successfully!")
            top.destroy()
            if not messagebox.askyesno("Flappy Bird", "发送成功！是否继续发送？"):
                self.root.destroy()
        except Exception as e:
            logging.error("%s" % e)
            messagebox.showerror("Flappy Bird", "%s" % e)


if __name__ == '__main__':
    Email("", 0)
