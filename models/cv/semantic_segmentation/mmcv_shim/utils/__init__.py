"""mmcv.utils shim."""

from mmcv_shim._impl import (  # noqa: F401
    Registry,
    build_from_cfg,
    deprecated_api_warning,
    digit_version,
    get_logger,
    is_list_of,
    is_str,
    is_tuple_of,
    print_log,
)
