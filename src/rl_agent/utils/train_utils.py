import copy
import traceback
import warnings

from ray.tune.logger import logger
from typing import Dict


class TrainUtils:
    @staticmethod
    def print_warning_tracebacks():
        def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
            wrn = "%s: %s\n" % (category.__name__, message)
            logger.warning(wrn + ''.join(traceback.format_stack()[:-2]) + wrn)

        warnings.showwarning = warn_with_traceback

    @staticmethod
    def with_base_config(base_config: Dict, extra_config: Dict) -> Dict:
        """Returns the given config dict merged with a base agent conf."""

        config = copy.deepcopy(base_config)
        config.update(extra_config)

        return config