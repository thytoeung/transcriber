# Transcriber

A local, GPU-accelerated transcription app for brainstorms and client discovery sessions. Audio and transcripts never leave the laptop.

**Pipeline:**
1. **whisperX** transcribes with faster-whisper (large-v3 by default, English).
2. wav2vec2 aligns each word to the audio.
3. pyannote identifies speakers (optional).
4. **Ollama** (qwen3 / llama3.1) writes the summary (optional).

## Run it

Double-click **`Transcriber.pyw`**, or run `python -m transcript_app` from this folder.

## One-time setup for speaker identification

1. Create a free account at https://huggingface.co.
2. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1.
3. Create a **Read** token at https://huggingface.co/settings/tokens.
4. In the app, open **Settings** (bottom left), paste the token, and click **Save & test access**.

The speaker model downloads once and then runs offline.

## Using it

1. **Add recordings.** Drop audio or video files on the window, or click *Add files…*. Supported formats include mp3, m4a, wav, flac, mp4 and mov.
2. **Pick a preset:**
   - **Solo brainstorm:** no speaker identification; brainstorm-style summary.
   - **Client session:** speaker identification on; discovery-notes summary.

   Setting **Speakers** to the exact number of people (instead of Auto-detect) improves speaker accuracy.
3. **Pick custom vocabulary.** Each file uses up to three lists:
   - **`default`** is always on.
   - Up to two optional lists, such as one per client.

   Each list is a plain `.txt` file with one term per line (or comma-separated); lines starting with `#` are ignored. Use the buttons to manage them:
   - **Open** edits a list in your default text editor.
   - **New list…** creates one.
   - **Folder** shows where the lists live: `%APPDATA%\Transcriber\vocabulary`, outside this repo, so client names are never committed.

   Lists are read when each file starts transcribing, so edits made while a batch is queued are picked up. Whisper only reads roughly the last 200 words of the combined vocabulary, so keep lists focused. The log warns you when they're too long.
4. **Choose what happens when it's finished:** the output folder, the export formats (TXT, MD, DOCX, SRT, VTT), and whether to summarize with a local LLM.
5. Click **Transcribe**. Files are processed in order, so you can queue a batch and walk away.

Settings in the **Transcription** box are captured when you add each file. Settings in **When finished** apply to the whole queue.

## Reviewing a transcript

Double-click a finished file in the queue to open it.

- **Show: Timecodes / Speaker names** turn those on or off in the view *and* in exports.
- Click a **▶ timecode** to play from that point. *Follow playback* highlights the paragraph being played.
- **Rename speakers** in the bar above the table (for example, "Speaker 1" → "Thy").
- Double-click **text** to fix a word. Double-click a **speaker** to reassign that paragraph.
- Right-click a paragraph to merge it with the one before or after.
- **Ctrl+F** finds text, **Ctrl+Space** plays or pauses, and **Ctrl+S** saves.
- **Export** rewrites the ticked formats with your edits.
- The **Summary** tab generates or regenerates the summary, and lets you edit it as Markdown.

## Output files

The project file is `<name>.transcript.json`. It holds every word with its timing and speaker, plus your edits and the summary. Exports sit next to it as `<name>.transcript.txt`, `.md`, `.docx`, `.srt` and `.vtt`. Reopen the `.json` file anytime to edit and export again without re-transcribing.

## Models

| Model | Speed | Notes |
|---|---|---|
| large-v3 (default) | Baseline | Best accuracy |
| large-v3-turbo | about 3× faster | Nearly the same accuracy |
| distil-large-v3 | Fast | Slightly less robust |
| medium.en / small.en | Fastest | For drafts |

## Layout

| Path | Contents |
|---|---|
| `transcript_app/engine.py` | GPU worker (separate process, killed on Cancel, frees all VRAM on exit). Can also be run from the command line: `python -m transcript_app.engine job.json` |
| `transcript_app/document.py` | Transcript model: words → speaker-smoothed paragraphs, JSON save/load |
| `transcript_app/exporters.py` | TXT, MD, DOCX, SRT and VTT export |
| `transcript_app/summarize.py` | Summary prompts. Long recordings are summarized in chunks, then merged. |
| `transcript_app/gui.py`, `viewer.py`, `player.py`, `runner.py` | PySide6 UI |
| `C:\aiprojects\shared\localllm` | Shared Ollama client, reusable by other projects |

## Troubleshooting

- **GPU out of memory:** the engine retries automatically with smaller batches. If it still fails, make sure no Ollama model is loaded (`ollama ps`), or choose large-v3-turbo.
- **Speaker identification refused:** accept the model terms on Hugging Face (step 2 above) with the same account that owns the token.
- **Audio decoding:** if `ffmpeg` is on PATH it's used to decode audio; otherwise PyAV decodes in-process. Either works.
