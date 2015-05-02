#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Zoa Chou'
# see http://www.mudoom.com/Article/show/id/30.html for detail

import sys
import os
import logging
from factory import make_celery
from mailer import Mailer, MailerException
from celery.exceptions import MaxRetriesExceededError
sys.path.append(os.path.realpath(os.path.realpath(__file__)+'/../../'))
celery = make_celery()


@celery.task(name='mail',
             bind=True,
             default_retry_delay=celery.conf.get('CELERY_RETRY_DELAY'),
             max_retries=celery.conf.get('CELERY_MAX_RETRIES'))
def mail(self, sender, content):
    smtp_server = {'host': celery.conf.get('MAIL_SMTP_SERVER'),
                   'port': celery.conf.get('MAIL_SMTP_PORT'),
                   'user': celery.conf.get('MAIL_SMTP_USER'),
                   'passwd': celery.conf.get('MAIL_SMTP_PASSWORD'),
                   'ssl': celery.conf.get('MAIL_SMTP_SSL')}
    mailer = Mailer()

    try:
        mailer.send_mail(
            smtp_server,
            celery.conf.get('MAIL_FROM_ADDRESS'),
            sender,
            celery.conf.get('MAIL_SUBJECT'),
            content,
        )

    except MailerException, e:
        logging.warning(e)
        try:
            self.retry()
        except MaxRetriesExceededError:
            """ 重试次数超过后依然失败的处理 """
            pass

    except Exception, e:
        logging.error(e)


if __name__ == '__name__':
    sender = 'email for who your want to send'
    content = 'input your content'
    mail_level = 3

    # 根据需求启用不同等级QUEUE
    celery_queue = {
        1: 'mail_fastest',
        2: 'mail_faster',
        3: 'default',
        4: 'mail_slower',
        5: 'mail_slowest'
    }
    current_queue = celery_queue.get(mail_level, 'default')
    mail.apply_async(args=[sender, content], queue=current_queue)
    pass
    # run with celery worker -A tasks -Q default for mail
    # run with celery worker -A tasks -Q mail_fastest for mail
    # run with celery worker -A tasks -Q mail_faster for mail
    # run with celery worker -A tasks -Q mail_slower for mail
    # run with celery worker -A tasks -Q mail_slowest for mail