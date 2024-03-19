import os

from faster_whisper import WhisperModel
from loguru import logger
from openai import OpenAI

import record

if __name__ == "__main__":
    model_size = "small"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_API_BASE_URL = os.environ.get(
        "OPENAI_API_BASE_URL", "https://api.perplexity.ai"
    )
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "sonar-small-online")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL)

    recording = record.Recording()
    recording.run()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, concise, polite conversation with a user."
                "The user lives in Spain."
            ),
        }
    ]

    logger.info("Searching for utterances")
    while True:
        utterance = recording.get_utterance()
        if utterance.getvalue():
            logger.info("Transcribing utterance")
            try:
                segments, info = model.transcribe(utterance, beam_size=5)
                for segment in segments:
                    logger.debug(
                        "[%.2fs -> %.2fs] %s"
                        % (segment.start, segment.end, segment.text)
                    )
                    logger.info("user: %s " % segment.text)
                    messages.append({"role": "user", "content": segment.text})
                    response = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=messages,
                    )
                    response_content = response.choices[0].message.content
                    logger.info("gecko: %s" % response_content)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_content,
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to transcribe: {e}")
            recording.clear_utterance()
    logger.info("Finished transcribing")
