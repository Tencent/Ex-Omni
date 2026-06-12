from .speech_encoder import HFSpeechEncoder


def build_speech_encoder(config):
    return HFSpeechEncoder.load(config)
