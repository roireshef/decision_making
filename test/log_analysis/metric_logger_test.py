import pytest
from time import sleep
import os
from pathlib import Path
from decision_making.src.utils.metric_logger import MetricLogger


"""
!!IMPORTANT!!! 

Testcase writes multiple lines to the same log so all test_XX functions  are dependent (by order) 
ideally, this should be implemented with sending the fixture 'rm_logflie' as an argument to each test function
and calling AV_Logger.shutdown_logger() after every call. 

"""

def get_log_file():
    root=os.path.join(os.path.dirname(__file__)).split('spav')[0]
    return root + 'spav/logs/AV_Log_JSON__jb_pytest_runner.log'

@pytest.fixture
def rm_logfile():
    try:
        os.remove(get_log_file())
    except OSError:
        pass

def checkFile():
    print('log_file:'+str(Path(get_log_file()).exists()))

def get_codelines():
    sleep(1) #waiting for logger to end
    with open(get_log_file(), 'r') as log_file_recs:
        return [eval(ll.split('data:')[1].replace('\'','"').rstrip()[:-1]) for ll in log_file_recs if ll.find('TEST')]


def test_MetricLogger_simpleoutput_OutputsToLogFile():
    simple_message= 'Just a simple message'
    try:
        MetricLogger.init('TEST')
        logger = MetricLogger.get_logger()
        logger.report(simple_message)

    except Exception as e:
        pytest.fail("Exception was thrown", e)
    ll= get_codelines()[0]
    assert ll['message'] == simple_message
    #AV_Logger.shutdown_logger()



def test_MetricLogger_withAutobinding_correctReport():
    binded_message = 'Message with args: %d and binded data'
    try:

        MetricLogger.init('TEST')
        logger = MetricLogger.get_logger()
        logger.report(binded_message, 3, a='arg1', b='arg2')
    except:
        pytest.fail("Exception was thrown")
    assert get_codelines()[1]['message'] == binded_message % 3
    assert get_codelines()[1]['TEST_a'] == 'arg1'
    assert get_codelines()[1]['TEST_b'] == 'arg2'


def test_MetricLogger_simpleBinding_correctReport():
    try:
        MetricLogger.init('TEST')
        logger = MetricLogger.get_logger()

        logger.bind(data={'x': 10})
        logger.report()
    except:
        pytest.fail("Exception was thrown")
    assert get_codelines()[2]['TEST_data'] == {'x':10}


def test_MetricLogger_multipleMessagesSingleBinding_correctReport():
    try:
        MetricLogger.init('TEST')
        logger = MetricLogger.get_logger()
        logger.bind(a=1, b=2)
        logger.report('just a message')
        logger.report('just another message 1')
    except:
        pytest.fail("Exception was thrown")
    assert get_codelines()[3]['message'] == 'just a message'
    assert get_codelines()[3]['TEST_a'] == 1
    assert get_codelines()[3]['TEST_b'] == 2

    assert get_codelines()[4]['message'] == 'just another message 1'
    assert get_codelines()[4]['TEST_a'] == 1
    assert get_codelines()[4]['TEST_b'] == 2



def test_MetricLogger_bindingPersistencyOverCalls():
    try:
        MetricLogger.init('TEST')
        logger = MetricLogger.get_logger()
        logger.report('message should include a,b')
    except:
        pytest.fail("Exception was thrown")
    assert get_codelines()[5]['message'] == 'message should include a,b'
    assert get_codelines()[5]['TEST_a'] == 1
    assert get_codelines()[5]['TEST_b'] == 2


def test_MetricLogger_unbinding():
    try:
        MetricLogger.init('TEST')
        logger = MetricLogger.get_logger()
        logger.bind(a='a', b='b')
        logger.report('initial binding')
        logger.report('message with binding')
        logger.unbind('a', 'b')
        logger.report('without binding')
    except:
        pytest.fail("Exception was thrown")

    assert get_codelines()[6]['message'] == 'initial binding'
    assert get_codelines()[6]['TEST_a'] == 'a'
    assert get_codelines()[6]['TEST_b'] == 'b'

    assert get_codelines()[7]['TEST_a'] == 'a'
    assert get_codelines()[7]['TEST_b'] == 'b'

    assert get_codelines()[8]['message'] == 'without binding'
    assert 'TEST_a' not in get_codelines()[8]
    assert 'TEST_b' not in get_codelines()[8]
