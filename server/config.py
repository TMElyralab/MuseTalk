import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
class Config(BaseSettings):
    ENV: str = "development"
    DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 7999
    MUSETALK_PATH: str = os.getenv("MUSETALK_PATH")

class LocalConfig(Config):
    ...


class ProductionConfig(Config):
    DEBUG: bool = False


def get_config():
    env = os.getenv("ENV", "local")
    config_type = {
        "local": LocalConfig(),
        "prod": ProductionConfig(),
    }
    return config_type[env]

config: Config = get_config()
print("config", config)