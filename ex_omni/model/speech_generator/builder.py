from .speech_generator import SpeechGenerator


def build_speech_generator(config):
    return SpeechGenerator(config)
