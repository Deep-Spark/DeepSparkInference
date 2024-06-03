# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import argparse
import tensorrt
from tensorrt import Dims

def make_parser():
    parser = argparse.ArgumentParser("DBnet Build engine")
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--engine", default="", type=str,help="float16 None,int8 quant json file")
    return parser

def main(config):
    
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    precision =  tensorrt.BuilderFlag.FP16
    parser.parse_from_file(config.model)
    build_config.set_flag(precision)

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)

if __name__ == "__main__":
    config = make_parser().parse_args()
    main(config)
    
    
    
        
    

    
    
    
    
    

    

    
    