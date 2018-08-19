#!/bin/bash 
LOG_LOCATION=${HOME}/av_code/spav/logs/
LOG_NAME=AV_Log_JSON_run_dm_mains.log
TARGET_FILEBEAT_FILE=$LOG_LOCATION/metric_DM_planning.log

python "${HOME}/av_code/spav/decision_making/test/log_analysis/convert_to_valid_jsons.py" $LOG_LOCATION/$LOG_NAME > ${TARGET_FILEBEAT_FILE}

 
