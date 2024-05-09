import logging

from classification_model.config.core import PACKAGE_ROOT, config

logging.getLogger(config.app.package_name).addHandler(logging.NullHandler())

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
