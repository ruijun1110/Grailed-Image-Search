import logging
import json
from logging.handlers import QueueHandler
import queue


class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, self.datefmt),
        }
        return json.dumps(log_data)


class JSONQueueHandler(QueueHandler):
    def prepare(self, record):
        # Ensure the record is formatted as JSON before putting it in the queue
        return self.format(record)


def setup_logger(name, log_queue):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    queue_handler = JSONQueueHandler(log_queue)
    queue_handler.setFormatter(CustomFormatter())
    logger.addHandler(queue_handler)

    return logger
