from .speech_projector import EncoderProjectorConcat


def build_speech_projector(config):
    return EncoderProjectorConcat(config)
