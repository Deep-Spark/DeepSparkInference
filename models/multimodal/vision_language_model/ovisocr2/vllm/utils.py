# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import base64
import mimetypes
import os


def image_payload(source):
    """A local page scan has to be inlined; an http(s) address is passed through.

    Shared by the offline and online scripts so the two cannot drift apart.
    """
    if source.startswith(("http://", "https://", "data:")):
        return source
    if not os.path.isfile(source):
        raise SystemExit(f"image not found: {source}")
    mime = mimetypes.guess_type(source)[0] or "image/png"
    with open(source, "rb") as handle:
        return f"data:{mime};base64," + base64.b64encode(handle.read()).decode()
