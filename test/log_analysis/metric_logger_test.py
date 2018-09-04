import pytest
from time import sleep
import os
import glob
from pathlib import Path
from decision_making.src.utils.metric_logger import MetricLogger
from rte.python.logger.AV_logger import AV_Logger, LOG_DIRECTORY


def get_log_file():
    list_of_files = glob.glob(LOG_DIRECTORY + '/*JSON*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_logger_for_testing():
    """
    This is a hack/workaround for testing because the AV_Logger.shutdown_logger() works only
    before calling AV_Logger.get_JSON_logger().
    :return:
    """
    MetricLogger._instance = None
    MetricLogger.init('TEST')
    logger = MetricLogger.get_logger()
    return logger


@pytest.fixture
def rm_logfile():
    try:
        os.remove(get_log_file())
    except OSError:
        pass


def checkFile():
    print('log_file:' + str(Path(get_log_file()).exists()))


def get_codelines():
    sleep(1)  # waiting for logger to end
    with open(get_log_file(), 'r') as log_file_recs:
        return [eval(ll.split('data:')[1].replace('\'', '"').rstrip()[:-1]) for ll in log_file_recs if
                ll.find('TEST') != -1]


def test_MetricLogger_simpleoutput_OutputsToLogFile():
    simple_message = 'TEST Just a simple message'
    try:
        logger = get_logger_for_testing()
        logger.report(simple_message)
    except Exception as e:
        pytest.fail("Exception was thrown", e)
    ll = get_codelines()[0]
    print(get_codelines())
    assert ll['message'] == simple_message
    AV_Logger.shutdown_logger()
    sleep(2)


def test_MetricLogger_withAutobinding_correctReport():
    binded_message = 'TEST Message with args: %d and binded data'
    try:
        logger = get_logger_for_testing()
        logger.report(binded_message, 3, a='arg1', b='arg2')
    except:
        pytest.fail("Exception was thrown")
    print(get_codelines())
    assert get_codelines()[0]['message'] == binded_message % 3
    assert get_codelines()[0]['TEST_a'] == 'arg1'
    assert get_codelines()[0]['TEST_b'] == 'arg2'
    AV_Logger.shutdown_logger()
    sleep(1)


def test_MetricLogger_simpleBinding_correctReport():
    try:
        logger = get_logger_for_testing()
        logger.bind(data={'x': 10})
        logger.report()
    except:
        pytest.fail("Exception was thrown")
    print(get_codelines())
    assert get_codelines()[0]['TEST_data'] == {'x': 10}
    AV_Logger.shutdown_logger()
    sleep(1)


def test_MetricLogger_multipleMessagesSingleBinding_correctReport():
    try:
        logger = get_logger_for_testing()
        logger.bind(a=1, b=2)
        logger.report('TEST just a message')
        logger.report('TEST just another message 1')
    except:
        pytest.fail("Exception was thrown")
    assert get_codelines()[0]['message'] == 'TEST just a message'
    assert get_codelines()[0]['TEST_a'] == 1
    assert get_codelines()[0]['TEST_b'] == 2

    assert get_codelines()[1]['message'] == 'TEST just another message 1'
    assert get_codelines()[1]['TEST_a'] == 1
    assert get_codelines()[1]['TEST_b'] == 2

    AV_Logger.shutdown_logger()
    sleep(1)


def test_MetricLogger_unbinding():
    try:
        logger = get_logger_for_testing()
        logger.bind(a='a', b='b')
        logger.report('TEST initial binding')
        logger.report('TEST message with binding')
        logger.unbind('a', 'b')
        logger.report('TEST without binding')
    except:
        pytest.fail("Exception was thrown")

    assert get_codelines()[0]['message'] == 'TEST initial binding'
    assert get_codelines()[0]['TEST_a'] == 'a'
    assert get_codelines()[0]['TEST_b'] == 'b'

    assert get_codelines()[1]['TEST_a'] == 'a'
    assert get_codelines()[1]['TEST_b'] == 'b'

    assert get_codelines()[2]['message'] == 'TEST without binding'
    assert 'TEST_a' not in get_codelines()[2]
    assert 'TEST_b' not in get_codelines()[2]
    AV_Logger.shutdown_logger()
    sleep(1)
