import pytest
from time import sleep
import os
import glob
from pathlib import Path
from decision_making.src.utils.metric_logger import MetricLogger
from rte.python.logger.AV_logger import AV_Logger, LOG_DIRECTORY


def get_log_file():
    list_of_files = glob.glob(LOG_DIRECTORY + '/*JSON*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file


def get_logger_for_testing():
    """
    This is a hack/workaround for testing because the AV_Logger.shutdown_logger() works only
    before calling AV_Logger.get_JSON_logger(). The shutdown_logger is needed for starting a new logfile for
    every test case
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

def print_log_content():
    print('*********LOG*********')
    with open(get_log_file(),'r') as lf:
        for l in lf:
            print (l)
    print('********END-LOG*********')

def eval_quote(line):
    """
    Extract from log when double-quotes exits
    :param line:
    :return:
    """
    return eval(line.replace('"','').split('data:')[1].rstrip()[:-1])

def eval_noquote(line):
    """
    Extract from log when double-quotes exits
    :param line:
    :return:
    """
    return eval(line.split('data:')[1].rstrip()[:-1])


def get_loglines():
    sleep(1)  # waiting for logger to end
    print_log_content()
    with open(get_log_file(), 'r') as log_file_recs:
        return [eval_quote(ll) for ll in log_file_recs if ll.find('TEST') !=-1 and ll.find('data') !=-1]

#{"levelname":"DEBUG", "timestamp":"2018-09-06 14:24:27.520", "process":"79626", "processName":"DM_process_create_behavioral_planner", "filename":"metric_logger.py", "lineno":"61", "function":"report", "data":"{'message': '', 'args': {}}"}


#def get_loglines():
#    sleep(1)  # waiting for logger to end
#    tmp=[]
#    with open(get_log_file(), 'r') as log_file_recs:
#        for ll in log_file_recs:
#            #print(ll)
#            if ll.find('TEST') !=-1 and ll.find('data:') !=-1:
#               # print('*',)
#                #print(eval(ll.split('data:')[1].replace('\'', '"').rstrip()[:-1]))
#                tmp.append(eval(ll.split('data:')[1].replace('\'', '"').rstrip()[:-1]))
#       # tmp=[eval(ll.split('data:')[1].replace('\'', '"').rstrip()[:-1]) for ll in log_file_recs if ll.find('TEST') != -1 and ll.find('data:') !=-1]
#       # tmp=[eval(ll.split('data:')[1].replace('\'', '"').rstrip()[:-1]) for ll in log_file_recs if ll.find('TEST') != -1 and ll.find('data:') !=-1]
#       # print(tmp)
#    return tmp


def test_MetricLogger_simpleoutput_OutputsToLogFile():
    simple_message = 'TEST Just a simple message'
    try:
        logger = get_logger_for_testing()
        logger.report(simple_message)
    except Exception as e:
        pytest.fail("Exception was thrown", e)
    ll = get_loglines()[0]
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
    assert get_loglines()[0]['message'] == binded_message % 3
    assert get_loglines()[0]['TEST_a'] == 'arg1'
    assert get_loglines()[0]['TEST_b'] == 'arg2'
    AV_Logger.shutdown_logger()
    sleep(1)


def test_MetricLogger_simpleBinding_correctReport():
    try:
        logger = get_logger_for_testing()
        logger.bind(data={'x': 10})
        logger.report()
    except:
        pytest.fail("Exception was thrown")
    assert get_loglines()[0]['TEST_data'] == {'x': 10}
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
    assert get_loglines()[0]['message'] == 'TEST just a message'
    assert get_loglines()[0]['TEST_a'] == 1
    assert get_loglines()[0]['TEST_b'] == 2

    assert get_loglines()[1]['message'] == 'TEST just another message 1'
    assert get_loglines()[1]['TEST_a'] == 1
    assert get_loglines()[1]['TEST_b'] == 2

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

    assert get_loglines()[0]['message'] == 'TEST initial binding'
    assert get_loglines()[0]['TEST_a'] == 'a'
    assert get_loglines()[0]['TEST_b'] == 'b'

    assert get_loglines()[1]['TEST_a'] == 'a'
    assert get_loglines()[1]['TEST_b'] == 'b'

    assert get_loglines()[2]['message'] == 'TEST without binding'
    assert 'TEST_a' not in get_loglines()[2]
    assert 'TEST_b' not in get_loglines()[2]
    AV_Logger.shutdown_logger()
    sleep(1)
