import io
import threading
import wave

import numpy as np
import pyaudio
from loguru import logger

import vad


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


class Recording:

    def __init__(
        self,
        format=pyaudio.paInt16,
        channels=1,
        sample_rate=16000,
        chunk=1600,
        samples_number=1536,
    ):
        self.format = format
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk = int(self.sample_rate / 10)
        self.samples_number = samples_number

        self.continue_recording = False

        self.audio = pyaudio.PyAudio()
        self.vad = vad.VAD(sample_rate=self.sample_rate)

        self.audio_buffer = []
        self.voice_probabilities_buffer = []
        self.utterance = []

        self.stream = self.get_audio_stream()

    def get_audio_stream(self):
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk,
        )
        return stream

    def detect_silence(self):
        margin = int(self.sample_rate / self.chunk)
        if len(self.voice_probabilities_buffer) < margin:
            return False
        if not all(prob < 0.5 for prob in self.voice_probabilities_buffer[-margin:]):
            return False
        logger.debug("Silence detected")
        return True

    def voice_in_buffer(self):
        if not any(prob > 0.5 for prob in self.voice_probabilities_buffer):
            return False
        logger.debug("Voice detected in buffer")
        return True

    def detect_utterance(self):
        if self.detect_silence() and not self.voice_in_buffer():
            self.clear_buffers()
        if not self.voice_in_buffer():
            return False
        if not self.detect_silence():
            return False
        logger.info("Utterance detected")
        return True

    def get_utterance(self):
        container = io.BytesIO()
        if len(self.utterance) == 0:
            return container
        with wave.open(container, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.utterance))
        container.seek(0)
        return container

    def trigger(self):
        self.save_utterance()
        self.clear_buffers()

    def save_utterance(self):
        logger.debug("Utterance saved")
        self.utterance = self.audio_buffer

    def clear_utterance(self):
        logger.debug("Utterance cleared")
        self.utterance = []

    def clear_buffers(self):
        logger.debug("Buffers cleared")
        self.audio_buffer = []
        self.voice_probabilities_buffer = []

    def stop(self):
        input("Press Enter to stop the recording:\n")
        self.continue_recording = False
        logger.info("Recording stopped")

    def start(self):

        self.continue_recording = True

        logger.info("Recording started")
        while self.continue_recording:
            audio_chunk = self.stream.read(self.samples_number)
            self.audio_buffer.append(audio_chunk)
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = int2float(audio_int16)
            new_confidence = self.vad.get_voice_probabilities(audio_float32)
            self.voice_probabilities_buffer.append(new_confidence)
            if self.detect_utterance():
                self.trigger()

    def run(self):
        stop_listener = threading.Thread(target=self.stop)
        stop_listener.start()
        start_listener = threading.Thread(target=self.start)
        start_listener.start()
