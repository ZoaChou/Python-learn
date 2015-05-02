#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Zoa Chou'
# see http://www.mudoom.com/Article/show/id/29.html for detail

import logging
import smtplib
import mimetypes
import socket
from email import encoders
from email.header import Header
from email.mime.text import MIMEText, MIMENonMultipart
from email.mime.base import MIMEBase
from email.utils import parseaddr, formataddr


class Mailer(object):
    def __init__(self):
        pass

    def send_mail(self, smtp_server, from_address, to_address, subject, body, files=None):
        """
        发送邮件主程序
        :param smtp_server: dict 邮件服务器设置
            :keyword  host: string smtp服务器地址
            :keyword  port: int smtp服务器端口号
            :keyword  user: string 用户名
            :keyword  passwd: string 密码
            :keyword  ssl: bool 是否启用ssl,默认False
            :keyword  timeout: int 超时时间,默认10s
        :param from_address: 发件人邮箱
        :param to_address: 收件人邮箱
        :param subject: 邮件标题
        :param body: 邮件内容
        :param files: 附件
        :raise: NetworkError/MailerException
        """
        # 格式化邮件内容
        body = self._encode_utf8(body)
        # 邮件类型
        content_type = 'html' if body.startswith('<html>') else 'plain'
        msg = MIMENonMultipart() if files else MIMEText(body, content_type, 'utf-8')
        # 格式化邮件数据
        msg['From'] = self._format_address(from_address)
        msg['To'] = ', '.join(self._format_list(to_address))
        msg['subject'] = self._encode_utf8(subject)

        # 构造附件数据
        if files:
            msg.attach(MIMEText(body, content_type, 'utf-8'))
            cid = 0
            for file_name, payload in files:
                file_name = self._encode_utf8(file_name)
                main_type, sub_type = self._get_file_type(file_name)
                if hasattr(payload, 'read'):
                    payload = payload.read()
                f_name = self._encode_header(file_name)
                mime = MIMEBase(main_type, sub_type, filename=f_name)
                mime.add_header('Content-Disposition', 'attachment', filename=f_name)
                mime.add_header('Content-ID', '<%s>' % cid)
                mime.add_header('X-Attachment-Id', '%s' % cid)
                mime.set_payload(payload)
                encoders.encode_base64(mime)
                msg.attach(mime)
                cid += 1

        host = smtp_server.get('host')
        port = smtp_server.get('port')
        user = smtp_server.get('user')
        passwd = smtp_server.get('passwd')
        ssl = smtp_server.get('ssl', False)
        time_out = smtp_server.get('timeout', 10)

        # 没有输入端口则使用默认端口
        if port is None or port == 0:
            if ssl:
                port = 465
            else:
                port = 25

        logging.debug('Send mail form %s to %s' % (msg['From'], msg['To']))

        try:
            if ssl:
                # 开启ssl连接模式
                server = smtplib.SMTP_SSL('%s:%d' % (host, port), timeout=time_out)
            else:
                server = smtplib.SMTP('%s:%d' % (host, port), timeout=time_out)
            # 开启调试模式
            # server.set_debuglevel(1)

            # 如果存在用户名密码则尝试登录
            if user and passwd:
                server.login(user, passwd)

            # 发送邮件
            server.sendmail(from_address, to_address, msg.as_string())

            logging.debug('Mail sent success.')

            # 关闭stmp连接
            server.quit()

        except socket.gaierror, e:
            """ 网络无法连接 """
            logging.exception(e)
            raise NetworkError(e)

        except smtplib.SMTPServerDisconnected, e:
            """ 网络连接异常 """
            logging.exception(e)
            raise NetworkError(e)

        except smtplib.SMTPException, e:
            """ 邮件发送异常 """
            logging.exception(e)
            raise MailerException(e)

    def _format_address(self, s):
        """
        格式化邮件地址
        :param s:string 邮件地址
        :return: string 格式化后的邮件地址
        """
        name, address = parseaddr(s)
        return formataddr((self._encode_header(name), self._encode_utf8(address)))

    def _encode_header(self, s):
        """
        格式化符合MIME的头部数据
        :param s: string 待格式化数据
        :return: 格式化后的数据
        """
        return Header(s, 'utf-8').encode()

    def _encode_utf8(self, s):
        """
        格式化成utf-8编码
        :param s: string 待格式化数据
        :return: string 格式化后的数据
        """
        if isinstance(s, unicode):
            return s.encode('utf-8')
        else:
            return s

    def _get_file_type(self, file_name):
        """
        获取附件类型
        :param file_name: 附件文件名
        :return: dict 附件MIME
        """
        s = file_name.lower()
        pos = s.rfind('.')
        if pos == -1:
            return 'application', 'octet-stream'

        ext = s[pos:]
        mime = mimetypes.types_map.get(ext, 'application/octet-stream')
        pos = mime.find('/')
        if pos == (-1):
            return mime, ''
        return mime[:pos], mime[pos+1:]

    def _format_list(self, address):
        """
        将收件人地址格式化成list
        :param address: string/list 收件人邮箱
        :return: list 收件人邮箱list
        """
        l = address
        if isinstance(l, basestring):
            l = [l]
        return [self._format_address(s) for s in l]


class MailerException(Exception):
    """ 邮件发送异常类 """
    pass


class NetworkError(MailerException):
    """ 网络异常类 """
    pass

# test for @qq.com
if __name__ == '__main__':
    import sys

    def prompt(prompt):
        """
        接收终端输入的数据
        """
        sys.stdout.write(prompt + ": ")
        return sys.stdin.readline().strip()

    from_address = prompt("From(Only @qq.com)")
    passwd = prompt("Password")
    to_address = prompt("To").split(',')
    subject = prompt("Subject")
    print "Enter message, end with ^D:"
    msg = ''
    while 1:
        line = sys.stdin.readline()
        if not line:
            break
        msg = msg + line
    print "Message length is %d" % len(msg)
    # QQ邮箱默认设置
    smtp_server = {'host': 'smtp.qq.com', 'port': None, 'user': from_address, 'passwd': passwd, 'ssl': True}
    mailer = Mailer()

    try:
        mailer.send_mail(smtp_server, from_address, to_address, subject, msg)
    except MailerException, e:
        print(e)