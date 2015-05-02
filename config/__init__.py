# -*- coding: utf-8 -*-

__author__ = 'Zoa Chou'

import os


def load_config():
    mode = os.environ.get('MODE')

    try:
        if mode == 'PRODUCTION':
            from .production import ProductionConfig
            return ProductionConfig
        elif mode == 'TEST':
            from .testing import TestConfig
            return TestConfig
        else:
            from .development import DevelopmentConfig
            return DevelopmentConfig
    except ImportError, e:
        from .default import Config
        return Config