from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_json_metadata(temporary: Path, path: Path) -> None:
    """Bounded handling of transient Windows sharing/access collisions.

    Readers can briefly open status JSON without delete sharing, which prevents
    atomic replace on Windows. This retries only this metadata syscall, not an
    experiment or training step. Permanent access denial still raises after at
    most eight attempts (0.71 seconds of total backoff); no ACL is changed.
    """
    delays = (0.01, 0.02, 0.04, 0.08, 0.16, 0.20, 0.20)
    for attempt in range(len(delays) + 1):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            if attempt == 0:
                warnings.warn(f"Transient Windows JSON metadata replacement collision for {path.name}; "
                              "bounded syscall backoff, no experiment restart", RuntimeWarning)
            time.sleep(delays[attempt])


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _replace_json_metadata(temporary, path)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
