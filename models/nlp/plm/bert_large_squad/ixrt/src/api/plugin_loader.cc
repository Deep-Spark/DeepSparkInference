/* Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
* All Rights Reserved.
*
*    Licensed under the Apache License, Version 2.0 (the "License"); you may
*    not use this file except in compliance with the License. You may obtain
*    a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*    License for the specific language governing permissions and limitations
*    under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <memory>
#include <mutex>
#include <stack>
#include <unordered_set>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "custom_fc/fcPlugin.h"
#include "emb_layernorm/embLayerNormPlugin.h"
#include "emb_layernorm/embLayerNormInt8Plugin.h"
#include "gelu/geluPlugin.h"
#include "qkv_to_context/qkvToContextInt8Plugin.h"
#include "qkv_to_context/qkvToContextPlugin.h"
#include "skip_layernorm/skipLayerNormInt8Plugin.h"
#include "skip_layernorm/skipLayerNormPlugin.h"
#include "ffn/ffnPlugin.h"

using namespace nvinfer1;

namespace nvinfer1 {
namespace ixrt_plugin {

extern ILogger* gLogger;

}  // namespace plugin
}  // namespace nvinfer1

namespace {
// This singleton ensures that each plugin is only registered once for a given
// namespace and type, and attempts of duplicate registration are ignored.
class PluginCreatorRegistry {
   public:
    static PluginCreatorRegistry& getInstance() {
        static PluginCreatorRegistry instance;
        return instance;
    }

    string GetPluginUniqKey(const AsciiChar* const plugin_namespace, const AsciiChar* const plugin_name,
                            const AsciiChar* const plugin_version) {
        stringstream os;
        if (plugin_namespace[0] != '\0') {
            os << plugin_namespace << "/";
        }
        os << plugin_name;
        if (plugin_version[0] != '\0') {
            os << "/" << plugin_version;
        }
        return os.str();
    }

    template <typename CreatorType>
    void addPluginCreator(void* logger, char const* libNamespace) {
        printf("start addPluginCreator %s\n", libNamespace);
        // Make accesses to the plugin creator registry thread safe
        std::lock_guard<std::mutex> lock(mRegistryLock);

        std::string errorMsg;
        std::string verboseMsg;

        std::unique_ptr<CreatorType> pluginCreator{new CreatorType{}};
        pluginCreator->setPluginNamespace(libNamespace);

        nvinfer1::ixrt_plugin::gLogger = static_cast<nvinfer1::ILogger*>(logger);
        std::string pluginType = GetPluginUniqKey(pluginCreator->getPluginNamespace(), pluginCreator->getPluginName(),
                                                  pluginCreator->getPluginVersion());

        if (mRegistryList.find(pluginType) == mRegistryList.end()) {
            bool status = getPluginRegistry()->registerCreator(*pluginCreator, libNamespace);
            if (status) {
                mRegistry.push(std::move(pluginCreator));
                mRegistryList.insert(pluginType);
                printf("Registered plugin creator -  %s\n", pluginType.c_str());
                verboseMsg = "Registered plugin creator - " + pluginType;
            } else {
                printf("Could not register plugin creator - %s\n", pluginType.c_str());
                errorMsg = "Could not register plugin creator -  " + pluginType;
            }
        } else {
            printf("Plugin creator already registered - %s\n", pluginType.c_str());
            verboseMsg = "Plugin creator already registered - " + pluginType;
        }

        if (logger) {
            if (!errorMsg.empty()) {
                nvinfer1::ixrt_plugin::gLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
            }
            if (!verboseMsg.empty()) {
                nvinfer1::ixrt_plugin::gLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
            }
        }
    }

    ~PluginCreatorRegistry() {
        std::lock_guard<std::mutex> lock(mRegistryLock);

        // Release pluginCreators in LIFO order of registration.
        while (!mRegistry.empty()) {
            mRegistry.pop();
        }
        mRegistryList.clear();
    }

   private:
    PluginCreatorRegistry() {}

    std::mutex mRegistryLock;
    std::stack<std::unique_ptr<IPluginCreator>> mRegistry;
    std::unordered_set<std::string> mRegistryList;

   public:
    PluginCreatorRegistry(PluginCreatorRegistry const&) = delete;
    void operator=(PluginCreatorRegistry const&) = delete;
};

template <typename CreatorType>
void initializePlugin(void* logger, char const* libNamespace) {
    PluginCreatorRegistry::getInstance().addPluginCreator<CreatorType>(logger, libNamespace);
}

}  // namespace

extern "C" {
bool initLibNvInferPlugins(void* logger, const char* libNamespace) {
    initializePlugin<nvinfer1::ixrt_plugin::bert::FCPluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::FCInt8PluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::FFNPluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::EmbLayerNormPluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::EmbLayerNormInt8PluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::GeluPluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::QKVToContextPluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::QKVToContextInt8PluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::SkipLayerNormPluginDynamicCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::ixrt_plugin::bert::SkipLayerNormInt8PluginHFaceCreator>(logger, libNamespace);
    return true;
}
}
