import pytest

import numpy as np

from decision_making.src.exceptions import TimeAlignmentPredictionHorizonTooLong
from decision_making.src.global_constants import TIMESTAMP_RESOLUTION_IN_SEC, TIME_ALIGNMENT_PREDICTOR_MAX_HORIZON
from decision_making.src.prediction.time_alignment_predictor import TimeAlignmentPredictor
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.custom_fixtures import state_with_old_object, UPDATED_TIMESTAMP_PARAM, \
    OLD_TIMESTAMP_PARAM

STATE_WITH_OLD_OBJECT_FIXTURE = 'state_with_old_object'

UPDATED_TIMESTAMP = 1.0 / TIMESTAMP_RESOLUTION_IN_SEC


@pytest.mark.parametrize(STATE_WITH_OLD_OBJECT_FIXTURE, [{UPDATED_TIMESTAMP_PARAM: UPDATED_TIMESTAMP,
                                                          OLD_TIMESTAMP_PARAM: UPDATED_TIMESTAMP - 1.1 * TIME_ALIGNMENT_PREDICTOR_MAX_HORIZON / TIMESTAMP_RESOLUTION_IN_SEC}],
                         indirect=True)
def test_predictState_horizonTooLong_raiseException(state_with_old_object):
    logger = AV_Logger.get_logger("test_predictState_precisePrediction")
    predictor = TimeAlignmentPredictor(logger)

    try:
        predicted_state = predictor.predict_state(state=state_with_old_object,
                                                  prediction_timestamps=np.array([UPDATED_TIMESTAMP*TIMESTAMP_RESOLUTION_IN_SEC]))
        assert False
    except TimeAlignmentPredictionHorizonTooLong:
        assert True
    except Exception:
        assert False


@pytest.mark.parametrize(STATE_WITH_OLD_OBJECT_FIXTURE, [{UPDATED_TIMESTAMP_PARAM: UPDATED_TIMESTAMP,
                                                          OLD_TIMESTAMP_PARAM: UPDATED_TIMESTAMP - 0.5 * TIME_ALIGNMENT_PREDICTOR_MAX_HORIZON / TIMESTAMP_RESOLUTION_IN_SEC}],
                         indirect=True)
def test_predictState_horizonNotTooLong_noException(state_with_old_object):
    logger = AV_Logger.get_logger("test_predictState_precisePrediction")
    predictor = TimeAlignmentPredictor(logger)

    try:
        predicted_state = predictor.predict_state(state=state_with_old_object,
                                                  prediction_timestamps=np.array([UPDATED_TIMESTAMP*TIMESTAMP_RESOLUTION_IN_SEC]))
        assert True
    except Exception:
        assert False
