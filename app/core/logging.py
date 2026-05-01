import logging
import sys


logger = logging.getLogger("mica")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
