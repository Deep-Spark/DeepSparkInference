#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

IFS=","
read -r server_args client_args <<< "$@"
IFS=" "

python3 -m vllm.entrypoints.openai.api_server $server_args &

index=0
port="12345"
IFS=' ' read -r args <<< "$server_args"
arguments=($args)
for argument in $args
do
    index=`expr $index + 1`
    case $argument in
      --port) port=${arguments[index]};;
    esac
done

cleanup() {
    serve_pid=$(ps -ef | grep "vllm.entrypoints.openai.api_server" | grep "$port" | awk '{print $2}' | grep -v "grep")
    # first try kill -15 to serve pid
    # kill -15 "$serve_pid"
    while read pid; do
        kill -15 $pid
        sleep 5
    done <<< $serve_pid
    sleep 20
    main_work_pid=$(ps -ef | grep "$serve_pid" | awk '{print $2}' | grep -v "grep" || echo "")
    while read pid; do
        if [ -n "$( ixsmi | grep "$pid" )" ];then
            kill -9 $pid
            sleep 5
        fi
    done <<< $main_work_pid
}

trap cleanup EXIT

sleep_time=20
status=1
duration_times=0
while [ $status -ne 0 ]; do
    sleep $sleep_time
    duration_times=$(expr $duration_times + 1)
    if ! timeout 3 bash -c "</dev/tcp/127.0.0.1/$port" >/dev/null 2>&1; then
        status=1
        if [ $duration_times -gt 500 ]; then
            echo "connection time out, the port may can not be used, closing ..."
            exit 1
        fi
    else
        status=0
    fi
done

python3 "benchmark_serving.py" $client_args

exit $?