from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_WEIGHTS: str = "best_modsim.torchscript"

    model_config = SettingsConfigDict(env_prefix="SATDET_", case_sensitive=False)


settings = Settings()
