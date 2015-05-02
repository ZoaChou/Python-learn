# -*- coding: utf-8 -*-

__author__ = 'Zoa Chou'

import sys
import os
from flask import Flask
from celery import Celery
sys.path.append(os.path.realpath(os.path.realpath(__file__)+'/../../../'))
from config import load_config

config = load_config()


def make_app():
    app = Flask(__name__)
    app.config.from_object(config)
    return app


def make_celery(app=None):
    app = app or make_app()
    celery = Celery(__name__, broker=app.config.get('CELERY_BROKER'))
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    celery.app = app
    return celery