from config.configs import ModelConfig, SFTConfig


def test_model_config_default_medgemma():
    config = ModelConfig()
    assert config.model_name == "google/medgemma-1.5-4b-it"


def test_sft_config_validates_positive_epochs():
    config = SFTConfig(num_epochs=1)
    assert config.num_epochs == 1
