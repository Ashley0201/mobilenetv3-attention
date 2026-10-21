"""项目基础模块测试：包可导入、版本号合法。"""

from __future__ import annotations

import re

import mobilenet_attention
from mobilenet_attention import utils


def test_package_importable():
    assert mobilenet_attention.__version__


def test_version_format():
    assert re.fullmatch(r"\d+\.\d+\.\d+", mobilenet_attention.__version__)


def test_utils_subpackage_importable():
    assert utils.__name__ == "mobilenet_attention.utils"
