import os
from pathlib import Path

class PathConfig():
    HOME_DIR: str = Path.home()
    SIMPLE_FACE_HOME: str = os.getenv(
            'SIMPLEFACE_HOME', 
            default=os.path.join(HOME_DIR, '.simple-face')
        )
    WEIGHTS_DIR: str = os.path.join(SIMPLE_FACE_HOME, 'weights')


class DownloadConfig():
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    DOWNLOAD_TIMEOUT = 10
