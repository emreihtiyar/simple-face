import os
import abc

from simple_face.commons import PathConfig as PCfg
from simple_face.commons.downloader import download_single_file

class BaseModel(metaclass=abc.ABCMeta):
    WEIGHTS_NAME = None
    WEIGHTS_URL = None
    _model = None
    _instance = None
    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, **kwargs)
        return cls._instance
    @classmethod
    def _check_weight(cls)->bool:
        if cls.WEIGHTS_NAME is None or cls.WEIGHTS_URL is None:
            assert False, "You must define WEIGHTS_NAME and WEIGHTS_URL"
        if not os.path.exists(os.path.join(PCfg.WEIGHTS_DIR, cls.WEIGHTS_NAME)):
            return download_single_file(cls.WEIGHTS_NAME, cls.WEIGHTS_URL, PCfg.WEIGHTS_DIR)
        return True
