import torch


class VAD:

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )
        # (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    def get_voice_probabilities(self, audio_float32):
        return self.model(torch.from_numpy(audio_float32), self.sample_rate).item()
