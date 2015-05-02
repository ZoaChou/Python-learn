# -*- coding: utf-8 -*-

__author__ = 'Zoa Chou'


class Config(object):
    # 邮件通知设置
    MAIL_SMTP_SERVER = 'smtp.qq.com'
    MAIL_SMTP_PORT = None
    MAIL_SMTP_USER = 'youremail@qq.com'
    MAIL_SMTP_PASSWORD = 'your email password'
    MAIL_SMTP_SSL = True
    MAIL_FROM_ADDRESS = 'Mailer admin<youremail@qq.com>'
    MAIL_SUBJECT = 'your email title'