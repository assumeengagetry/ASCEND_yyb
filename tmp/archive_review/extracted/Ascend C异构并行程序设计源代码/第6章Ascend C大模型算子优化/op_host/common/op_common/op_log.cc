/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include <stdio.h>
#include <vector>
#include <string.h>
#include <stdarg.h>
#include "util/op_log.h"

namespace opcommon {
#define OP_UT_LOG_PRINT(fmt)   \
  do {  \
    va_list args;  \
    va_start(args, fmt);  \
    vprintf(fmt, args);  \
    va_end(args);  \
    printf("\n");  \
  } while (false)

inline const string& GetLevelName(int logLevel)
{
    static const std::vector<std::string> gLogLevelName = {"DEBUG", "INFO", "WARN", "ERROR"};
    static const std::string unKnownName = "unknown";
    if (logLevel >= 0 && logLevel <static_cast<int64_t>(gLogLevelName.size())) {
        return gLogLevelName[logLevel];
    }

    return unKnownName;
}

inline const string& GetModuleName(int moduleId)
{
    static const std::string moduleName = "OP_PROTO";
    static const std::string unKnownModule = "unknown";
    if (moduleId == 1) {
        return moduleName;
    }

    return unKnownModule;
}

int OpCheckLogLevel(int moduleId, int logLevel)
{
    printf("moduleId is : [%d], logLevel is : [%d]", moduleId, logLevel);
    return 1;
}

void OpDlogInner(int moduleId, int level, const char *fmt, ...)
{
    printf("moduleId is : [%s], level is : [%s]", GetModuleName(moduleId).c_str(), GetLevelName(level).c_str());
    OP_UT_LOG_PRINT(fmt);
}
} // namespace opcommon