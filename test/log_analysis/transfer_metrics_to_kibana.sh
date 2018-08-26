#!/bin/bash
#
#  Process JSON logs and transfer it to kibana
#
#  Usage: ./transfer_metrics_to_kibana.sh <original_log> <target_file_suffix> [-s]
#  [-s] if no merge is required
#
#  Check:
#  1) Make sure the target file name is identical to what is described in filebeat.yml
#  2) Edit the index name in logstash-beats.conf
#
LOG_LOCATION=${HOME}/av_code/spav/logs
LOG_NAME=$1
TARGET_NAME=$2
SIMPLE=$3
TARGET_FILEBEAT_FILE="${LOG_LOCATION}/metric_DM_planning${TARGET_NAME}.log"
#python "${HOME}/av_code/spav/decision_making/test/log_analysis/convert_to_valid_jsons.py" $LOG_LOCATION/$LOG_NAME $3 > ${TARGET_FILEBEAT_FILE}
echo "${HOME}/av_code/spav/decision_making/test/log_analysis/convert_to_valid_jsons.py" $LOG_LOCATION/$LOG_NAME $3 to ${TARGET_FILEBEAT_FILE}

 
