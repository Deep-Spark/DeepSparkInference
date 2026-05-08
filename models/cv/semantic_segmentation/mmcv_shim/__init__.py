"""Minimal mmcv shim — replaces mmcv 1.x with plain cv2/numpy/stdlib."""

from mmcv_shim._impl import (  # noqa: F401
    Config,
    FileClient,
    Registry,
    build_from_cfg,
    clahe,
    bgr2hsv,
    hsv2bgr,
    lut_transform,
    DataContainer,
    collate,
    deprecated_api_warning,
    digit_version,
    get_dist_info,
    get_logger,
    imflip,
    imfrombytes,
    imread,
    imnormalize,
    impad,
    impad_to_multiple,
    imrescale,
    imresize,
    imresize_to_multiple,
    imrotate,
    is_list_of,
    is_str,
    is_tuple_of,
    list_from_file,
    mkdir_or_exist,
    print_log,
    scandir,
)

# Sub-namespaces (imported lazily to avoid circular)
from mmcv_shim import parallel   # noqa: F401
from mmcv_shim import runner     # noqa: F401
from mmcv_shim import utils      # noqa: F401
