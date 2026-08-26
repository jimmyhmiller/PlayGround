"""Application configuration."""

import os


class Config:
    """Runtime configuration for the server."""

    def __init__(self):
        self.timeout = 30
        self.max_retries = 3
        self.debug = False

    @classmethod
    def from_env(cls):
        config = cls()
        config.timeout = int(os.environ.get("APP_TIMEOUT", "30"))
        config.debug = os.environ.get("APP_DEBUG", "") == "1"
        return config
