# gecko

A self-hosted open-source voice assistant.

## Road map

- [x] Voice Activity Detection (VAD)
- [x] Text-to-Speech (TTS) using whisper
- [x] Perplexity AI responses
- [ ] Speech-to-Text (STT)

## Limitations
- [ ] No wake word detection
- [ ] Ctrl+C twice to stop the program

## Development

It is known to work on **python 3.11**.
You will need
- `python3.11-devel`
- `portaudio-devel`

**Debian**
```bash
sudo apt install python3.11-dev portaudio19-dev
```

**Fedora**
```bash
sudo dnf install python3.11-devel portaudio-devel
```


```bash
pip install -r requirements.txt
```

Configure your environment variables in a `.env` file.
```bash
cp .env.example .env
```
and **modify your values** to OpenAI or Perplexity keys.


```bash
cd src
python app.py
```
