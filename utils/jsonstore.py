# -*- coding: utf-8 -*-
"""Safe storage of JSON histories.

WHY. On 2026-09-06 at 05:04 UTC the experiment history was wiped entirely: there were
1529 points over three weeks, only what was recorded after that second remained.
The log showed the cause second by second:

    05:04:31  write point FET       (open("w") truncates the file, then 6.8 MB)
    05:04:32  outcome resolution    (reads the file AT THAT MOMENT -> a fragment)
    05:05:14  write point CRV       (writes over whatever was left)

Three things came together, harmless on their own:
  1. the write is NOT atomic: open(path,"w") truncates the file to zero, and for the whole
     duration of the write a fragment lies on disk;
  2. the read swallowed ANY error and returned {} — "no file" and "file
     corrupted" were indistinguishable;
  3. then came a write of the whole dictionary.

One unluckily coinciding second wiped the whole history, silently.

WHY NOT JUST os.replace. The first version of this module did the classic
atomic replace — and BROKE writing completely. In docker-compose these histories
are mounted ONE BY ONE (`./file.json:/app/file.json`), and renaming over
a mount point gives `OSError: [Errno 16] Device or resource busy`.
Caught by a check on the live container before it could do harm.

So there are two paths here:
  * where replacing is possible (a regular directory, running from the host) — os.replace,
    true atomicity;
  * where it is not (a mounted file) — writing IN PLACE, but the content
    is serialised to a string IN ADVANCE and written in ONE write call. The window during
    which an incomplete file lies on disk shrinks from the duration of a streaming
    7 MB write to a single system call.

TRUE atomicity will return if a DIRECTORY is mounted instead of individual
files. Until then — we hold a lock and a narrow window.

THE LOCK. Both bot jobs (point recording and outcome resolution) live in ONE
process, and it was they that collided. So an RLock on the module closes
the original incident completely: reads and writes no longer overlap. External
readers (scripts on the host) do not see the lock — the narrow window is for them.

An empty file is considered corrupted, not an empty history: save_json writes at
least "{}", so zero length means an interrupted write.
"""
import errno
import json
import os
import shutil
import threading
from datetime import datetime, timezone

_LOCK = threading.RLock()


class CorruptStoreError(RuntimeError):
    """The file exists but cannot be parsed. It MUST NOT be overwritten:
    an empty history would land on top of the corrupted file and the data would disappear."""


def load_json(path, default=None):
    with _LOCK:
        if not os.path.exists(path):
            return {} if default is None else default
        try:
            if os.path.getsize(path) == 0:
                raise ValueError("zero-length file — the write was interrupted")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            broken = f"{path}.corrupt-{stamp}"
            try:
                shutil.copy2(path, broken)
            except Exception:
                broken = "(could not make a copy)"
            raise CorruptStoreError(
                f"{path} cannot be parsed ({e}). Copy set aside: {broken}. "
                f"Writing aborted on purpose so as not to overwrite the history with an empty one."
            ) from e


def _write_in_place(path, payload):
    """Fallback path for a mounted file: a single write, with fsync."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def save_json(path, data, indent=2):
    with _LOCK:
        # Serialise BEFORE opening the file: if the data turns out to be unwritable,
        # the exception arrives before anything on disk is touched.
        payload = json.dumps(data, ensure_ascii=False, indent=indent)
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except OSError as e:
            if e.errno != errno.EBUSY:
                raise
        finally:
            # Otherwise the leftover from a failed replace stays forever.
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        _write_in_place(path, payload)
