from atlas.config import Settings


def test_defaults_when_no_env_file() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env == "dev"
    assert settings.log_level == "INFO"
    assert settings.port == 10000
