import pytest

from decision_making.src.utils.metric_logger import MetricLogger


def test_simpleoutput_notRasingException():
    try:
        logger = MetricLogger.get_logger('TEST')
        logger.report('Just a simple message')

    except Exception as e:
        pytest.fail("Exception was thrown",e)

def test_with_autobinding_notRasingException():
    try:
        logger = MetricLogger.get_logger('TEST')
        logger.report('Message with args: %d and binded data',3, a='arg1',b='arg2')

    except:
        pytest.fail("Exception was thrown")



def test_simplebinding_notRasingException():
    try:
        logger = MetricLogger.get_logger('TEST')
        logger.bind(data={'x': 10})
        logger.report()
    except:
        pytest.fail("Exception was thrown")

def test_multiple_messagesSingleBinding_notRasingException():
    try:
        logger = MetricLogger.get_logger('TEST')
        logger.bind(a=1, b=2, c=3, d=4)
        logger.report('just a message')
        logger.report('just another message 1')
        logger.report('just another message 2')
        #Manually check in logfile
    except:
        pytest.fail("Exception was thrown")

def test_binding_persistencyOverCalls():
    try:
        logger = MetricLogger.get_logger('TEST')
        logger.report('message should include a-d')
        #Manually check in logfile
    except:
        pytest.fail("Exception was thrown")

def test_unbinding_manualValidation_notRaisingException():
    try:
        logger = MetricLogger.get_logger('TEST')

        logger.bind(a='a', b='b', c='c', d='d')
        logger.report('initial binding')
        logger.report('message with binding')
        logger.report('message with binding 1')
        logger.report('message with binding 2')
        logger.unbind('a','b','c','d')
        logger.report('without binding')
        logger.report('without binding 1')
        logger.report('without binding 2')
    except:
        pytest.fail("Exception was thrown")

