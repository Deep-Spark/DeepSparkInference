# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


PIPCMD=pip3

## Install packages
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt update
    cmake_path=`command -v cmake`
    if [ -z "${cmake_path}" ]; then
        echo "Install cmake"
        apt install -y cmake
    fi
    unzip_path=`command -v unzip`
    if [ -z "${unzip_path}" ]; then
        echo "Install unzip"
        apt install -y unzip
    fi
    apt -y install libgl1-mesa-glx
    pyver=`python3 -c 'import sys; print(sys.version_info[:][0])'`
    pysubver=`python3 -c 'import sys; print(sys.version_info[:][1])'`
    apt -y install python${pyver}.${pysubver}-dev
elif [[ ${ID} == "centos" ]]; then
    cmake_path=`command -v cmake`
    if [ -z "${cmake_path}" ]; then
        echo "Install cmake"
        yum install -y cmake
    fi
    unzip_path=`command -v unzip`
    if [ -z "${unzip_path}" ]; then
        echo "Install unzip"
        yum install -y unzip
    fi
    yum -y install mesa-libGL
else
    cmake_path=`command -v cmake`
    if [ -z "${cmake_path}" ]; then
        echo "Install cmake"
        yum install -y cmake
    fi
    unzip_path=`command -v unzip`
    if [ -z "${unzip_path}" ]; then
        echo "Install unzip"
        yum install -y unzip
    fi
    yum -y install mesa-libGL
fi