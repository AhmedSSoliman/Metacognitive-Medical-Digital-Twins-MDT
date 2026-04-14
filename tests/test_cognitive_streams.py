from config.configs import CognitiveArchitectureConfig
from core.cognitive_streams import CognitiveStreamParser


SAMPLE_TEXT = """<think>Assessing likely pneumonia based on fever and productive cough.</think>
<patient_state>Potential lower respiratory infection with elevated risk factors.</patient_state>
<user_belief>User is anxious but cooperative and seeking clear next steps.</user_belief>
"""


def test_validate_with_details_returns_quality_info():
    parser = CognitiveStreamParser(CognitiveArchitectureConfig())
    streams = parser.parse(SAMPLE_TEXT)

    details = parser.validate_with_details(streams)

    assert details["is_valid"] is True
    assert "stream_validity" in details
    assert details["errors"] == []


def test_analyze_stream_quality_has_expected_keys():
    parser = CognitiveStreamParser(CognitiveArchitectureConfig())
    streams = parser.parse(SAMPLE_TEXT)

    quality = parser.analyze_stream_quality(streams)

    assert "think_reasoning_depth" in quality
    assert "patient_state_coding" in quality
    assert "user_belief_completeness" in quality
