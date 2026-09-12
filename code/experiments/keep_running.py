"""Keep a mac awake while an experiment runs.
"""

import os
import shutil
import subprocess
from contextlib import contextmanager

__all__ = ["keep_running"]


@contextmanager
def keep_running():
    """Prevent idle and system sleep for macOS
    """
    executable = shutil.which("caffeinate")
    if executable is None:
        yield
        return

    process = subprocess.Popen([executable, "-i", "-s", "-w", str(os.getpid())])
    try:
        yield
    finally:
        process.terminate()
        process.wait()
