import logging
import time
from importlib import resources

import oxiquizkit

start_time = time.time()


class RuntimeFormatter(logging.Formatter):
    def format(self, record):
        runtime_minutes = (time.time() - start_time) / 60.0
        record.runtime = f"{runtime_minutes:.2f}m"
        return super().format(record)


formatter = RuntimeFormatter(
    fmt="%(asctime)s - %(runtime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

file_handler = logging.FileHandler("quizkit.log")
stream_handler = logging.StreamHandler()

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)


def main():
    logger.info("Done.")


if __name__ == "__main__":
    main()
