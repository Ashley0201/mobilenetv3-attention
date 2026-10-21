"""断点续传下载 CIFAR-10 官方 Python 版本，并校验 MD5。

官方来源：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
用途：torchvision.datasets.CIFAR10(download=True) 检测到完整压缩包后直接解压。

用法：
    python scripts/download_cifar10.py
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "data" / "cifar-10-python.tar.gz"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, target: Path, retries: int = 50) -> None:
    """按 Range 续传；网络中断后重试，不删除已下载数据。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        offset = target.stat().st_size if target.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
                total = int(response.headers.get("Content-Length", 0)) + offset
                mode = "ab" if status == 206 and offset else "wb"
                if mode == "wb":
                    offset = 0
                print(
                    f"[{attempt}/{retries}] HTTP {status}, 从 {offset} 字节续传，"
                    f"总大小约 {total / 1024**2:.1f} MiB",
                    flush=True,
                )
                with target.open(mode) as f:
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        offset += len(chunk)
                return
        except Exception as exc:  # noqa: BLE001 - 网络异常需重试
            print(f"下载中断：{exc}；5 秒后重试", flush=True)
            time.sleep(5)
    raise RuntimeError("超过最大重试次数，下载未完成")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if TARGET.exists() and md5sum(TARGET) == MD5:
        print(f"已存在且 MD5 正确：{TARGET}")
        return 0

    download_with_resume(URL, TARGET)
    actual = md5sum(TARGET)
    if actual != MD5:
        raise RuntimeError(f"MD5 不匹配：期望 {MD5}，实际 {actual}")
    print(f"下载完成并通过 MD5 校验：{TARGET} ({TARGET.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
