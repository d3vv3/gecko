
from faster_whisper import WhisperModel
from loguru import logger

import record

model_size = "small"

if __name__ == "__main__":
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    recording = record.Recording()
    recording.run()
    logger.info("Searching for utterances")
    while True:
        utterance = recording.get_utterance()
        if utterance.getvalue():
            logger.info("Transcribing utterance")
            try:
                segments, info = model.transcribe(utterance, beam_size=5)
                for segment in segments:
                    print(
                        "[%.2fs -> %.2fs] %s"
                        % (segment.start, segment.end, segment.text)
                    )
            except Exception as e:
                logger.error(f"Failed to transcribe: {e}")
            recording.clear_utterance()
    logger.info("Finished transcribing")
