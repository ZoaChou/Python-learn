# -*- coding: utf-8 -*-

__author__ = 'Zoa Chou'
from kombu import Queue, Exchange


class Config(object):
    # 邮件通知设置
    MAIL_SMTP_SERVER = 'smtp.qq.com'
    MAIL_SMTP_PORT = None
    MAIL_SMTP_USER = 'youremail@qq.com'
    MAIL_SMTP_PASSWORD = 'your email password'
    MAIL_SMTP_SSL = True
    MAIL_FROM_ADDRESS = 'Mailer admin<youremail@qq.com>'
    MAIL_SUBJECT = 'your email title'

    # celery设置
    CELERY_BROKER = 'redis://localhost:6379/0'  # broker
    CELERY_RETRY_DELAY = 10  # celery每次重试的间隔时间
    CELERY_MAX_RETRIES = 5  # celery最大的重试测试
    default_exchange = Exchange('default', type='direct')
    mail_exchange = Exchange('mail', type='direct')
    # celery多队列设置
    CELERY_QUEUES = (
        Queue('default', default_exchange, routing_key='default'),
        Queue('mail_fastest', mail_exchange, routing_key='mail.fastest'),
        Queue('mail_faster', mail_exchange, routing_key='mail.faster'),
        Queue('mail_slower', mail_exchange, routing_key='mail.slower'),
        Queue('mail_slowest', mail_exchange, routing_key='mail.slowest'),
    )
    CELERY_DEFAULT_QUEUE = 'default'
    CELERY_DEFAULT_EXCHANGE = 'default'
    CELERY_DEFAULT_ROUTING_KEY = 'default'