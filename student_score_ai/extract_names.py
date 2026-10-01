"""Извлечение идентификационных фрагментов студентов из транскриптов."""

import os
import re
from collections.abc import Iterable
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class FoundedName(BaseModel):
    """Один фрагмент речи, который может идентифицировать студента.

    Attributes:
        name: дословный или искажённый фрагмент из распознавания речи.
        minute_start: номер минутного чанка, содержащего фрагмент.
        evidence: короткая цитата, подтверждающая наличие фрагмента.
        context: окружение фрагмента, нужное для последующей проверки.
    """

    name: str
    minute_start: int
    evidence: str = ""
    context: str = ""


class NameListWithTime(BaseModel):
    """Список идентификационных фрагментов с привязкой ко времени."""

    names: list[FoundedName]


class ExtractedIdentityFragment(BaseModel):
    """Структурированный ответ модели для одного фрагмента транскрипта.

    Attributes:
        fragment: фрагмент, похожий на фамилию, имя или их искажённое сочетание.
        evidence: точная короткая цитата из транскрипта.
        context: контекст, который объясняет, почему фрагмент относится к студенту.
    """

    fragment: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    context: str = Field(min_length=1)


class IdentityFragmentList(BaseModel):
    """Ответ извлекателя для одного минутного фрагмента записи."""

    fragments: list[ExtractedIdentityFragment]


# Один клиент нужен извлекателю всех минут одной лекции и повторно использует настройки среды.
llm = ChatOpenAI(
    model="gemma-4-26B-A4B-it",
    temperature=0.0,
    timeout=60,
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
)


def extract_student_names(path_transcript: Iterable[Path]) -> NameListWithTime:
    """Извлечь ФИО и искажённые идентификационные фрагменты студентов.

    Args:
        path_transcript: пути к минутным текстовым фрагментам лекции.

    Returns:
        Список фрагментов в хронологическом порядке с доказательством и контекстом.
    """
    # Нестрогое извлечение не теряет полезные ошибки ASR до сопоставления с реестром.
    extracted_names: list[FoundedName] = []
    structured_llm = llm.with_structured_output(IdentityFragmentList, strict=True)
    for path in tqdm(sorted(path_transcript, key=_minute_from_path)):
        transcript = path.read_text(encoding="utf-8")
        minute_start = _minute_from_path(path)
        result = structured_llm.invoke(_build_extraction_prompt(transcript))
        extracted_names.extend(
            _to_founded_name(fragment, minute_start)
            for fragment in result.fragments
        )

    # Повторы одной и той же реплики не дают лишних начислений и не скрывают повторные ответы.
    return NameListWithTime(names=_deduplicate_fragments(extracted_names))


def _build_extraction_prompt(transcript: str) -> str:
    """Сформировать инструкцию для поиска идентификационных фрагментов.

    Args:
        transcript: текст одного минутного чанка лекции.

    Returns:
        Инструкция для структурированного вызова языковой модели.
    """
    # Инструкция намеренно допускает ошибки ASR, чтобы поздний этап смог сверить их с реестром.
    return f"""
Ты анализируешь транскрипт лекции и ищешь фрагменты, которыми преподаватель
или студент идентифицирует студента.

Верни фрагмент, если он похож на фамилию, имя, полное ФИО или их искажённое
распознавание речи. Ищи также склеенные слова и неясные варианты, например
«Пиктимироводиана», «Прятого я» или «Горбачёв-Сеней». Не исправляй написание:
верни текст максимально близко к транскрипту.

Особенно важны фрагменты рядом со словами «фамилия», «имя», «назови»,
«кто отвечал», «ты говорил», «ещё раз», а также обращения преподавателя после
ответа студента. Не включай преподавателя, названия технологий, случайные
обычные слова и людей, которые не являются студентами занятия.

Для каждого фрагмента верни:
- fragment: сам фрагмент;
- evidence: короткая дословная цитата из текста с фрагментом;
- context: одна-две фразы, показывающие идентификационный контекст.

Транскрипт:
{transcript}
"""


def _minute_from_path(path: Path) -> int:
    """Извлечь номер минуты из имени файла чанка.

    Args:
        path: путь вида ``chunk_0007.txt``.

    Returns:
        Номер минутного чанка.

    Raises:
        ValueError: если имя файла не оканчивается номером чанка.
    """
    # Ошибка имени файла должна остановить конвейер: без времени нельзя безопасно проверить фрагмент.
    match = re.search(r"_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Не удалось определить минуту из имени чанка: {path.name}")
    return int(match.group(1))


def _to_founded_name(
    fragment: ExtractedIdentityFragment,
    minute_start: int,
) -> FoundedName:
    """Привязать результат извлечения к минуте исходного аудио.

    Args:
        fragment: фрагмент и его текстовое доказательство от модели.
        minute_start: минута чанка, из которого получен фрагмент.

    Returns:
        Унифицированная запись для последующей валидации.
    """
    # Очистка пробелов делает дедупликацию стабильной, не меняя написание, нужное для фонетического поиска.
    return FoundedName(
        name=" ".join(fragment.fragment.split()),
        minute_start=minute_start,
        evidence=" ".join(fragment.evidence.split()),
        context=" ".join(fragment.context.split()),
    )


def _deduplicate_fragments(fragments: list[FoundedName]) -> list[FoundedName]:
    """Убрать дубли одного фрагмента в одной минуте.

    Args:
        fragments: извлечённые фрагменты в хронологическом порядке.

    Returns:
        Фрагменты без точных нормализованных повторов.
    """
    # Дедупликация только внутри минуты сохраняет повторные ответы студента позднее в лекции.
    unique_fragments: list[FoundedName] = []
    seen: set[tuple[int, str]] = set()
    for fragment in fragments:
        key = (fragment.minute_start, fragment.name.casefold())
        if key not in seen:
            seen.add(key)
            unique_fragments.append(fragment)
    return unique_fragments
