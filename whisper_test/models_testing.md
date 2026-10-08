# Приоритет тестирования

## Qwen3-ASR-1.7B — главный кандидат для качества.
ollama pull frozenlab/qwen3-asr:1.7b
ollama run frozenlab/qwen3-asr:1.7b

## Qwen3-ASR-0.6B — компромисс между скоростью и качеством.
ollama pull frozenlab/qwen3-asr:0.6b
ollama run frozenlab/qwen3-asr:0.6b

- GigaAM v3 — специализированная русская модель.
- Phi-4-Multimodal — отдельный тест для задачи «аудио + инструкции + суммаризация».
- Moonshine — только если появится русская модель.
- Distil-Whisper — не включать в русское сравнение без специального multilingual checkpoint.

## Запуск через Ollama API

Скрипт использует OpenAI-совместимый endpoint Ollama
`/v1/audio/transcriptions`. Ollama на виртуальной машине проброшена локально
на `http://127.0.0.1:11434`; там доступны `frozenlab/qwen3-asr:0.6b` и
`frozenlab/qwen3-asr:1.7b`.

Пример для Qwen3-ASR-1.7B:

```bash
uv run python compare_noize_removing.py \
  whisper_test/v1/audio_test/chunk_0049_target_voice_focus.wav \
  whisper_test/v7/audio_test/chunk_0049_lr_60_40.wav \
  whisper_test/chunk_0049_original.wav \
  --backend ollama \
  --model frozenlab/qwen3-asr:1.7b \
  --ollama-base-url http://127.0.0.1:11434 \
  --output whisper_test/models_results/qwen3_asr_1_7b.csv \
  --details whisper_test/models_results/qwen3_asr_1_7b.txt
```

Для `frozenlab/qwen3-asr:0.6b` замените только значение `--model`. VAD и
метрики остаются теми же, что у Whisper, поэтому результаты сопоставимы.
