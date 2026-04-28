import yaml
from typing import Optional, Any
from pathlib import Path
from loguru import logger

class Config():

    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance =  super().__new__(cls)
            cls.instance._load_config()
        return cls.instance

    def _load_config(self,
                    path :Optional[str]= None):
        
        if path is None: 
            path = Path(__file__).resolve().parent/"default_config.yml"


        try :
            with open(path) as f : 
                self._dict = yaml.safe_load(f) or {}
        
        except FileNotFoundError:
            logger.error(f"Config file not found at {path}")
            self._dict = {}
        
        except yaml.YAMLError as e :
            logger.error(f" Error parsing the yaml file {e}")
            self._dict = {}

        
    def get(self, key : str , default: Any = None) -> Any: 
        return self._dict.get(key, default)
    
    @property
    def all (self):
        return self._dict