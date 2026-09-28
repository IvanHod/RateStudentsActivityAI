import os
import re
import unicodedata
from collections.abc import Iterable
from typing import Optional, Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from student_score_ai.extract_names import NameListWithTime

llm = ChatOpenAI(
    model="gemma-4-26B-A4B-it",
    temperature=0.1,
    max_retries=3,
    api_key=os.environ["LITELLM_API_KEY"],          # ключ LiteLLM proxy
    base_url=os.environ["LITELLM_BASE_URL"],        # например http://my-host:4000/v1
)

class ValidatedStudent(BaseModel):
    raw_name: str = Field(description="Имя/ФИО, извлеченное из транскрипта")
    status: Literal["valid", "uncertain", "invalid"] = Field(
        description="valid — уверенное совпадение, uncertain — похоже, но не точно, invalid — не найдено"
    )
    matched_name: Optional[str] = Field(
        default=None,
        description="Совпавшее имя из официального списка"
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Уверенность в совпадении от 0 до 100"
    )
    comment: str = Field(
        description="Короткое объяснение, почему принято такое решение"
    )
    minute: int = Field(description="Минута, когда студент говорил (передается во входных данных)")


class ValidationResult(BaseModel):
    results: list[ValidatedStudent]


def validate_student_names_with_gpt(
    found_names: NameListWithTime,
    allowed_names: Iterable[str],
) -> ValidationResult:
    """
    Валидирует найденные ФИО по официальному списку через GPT.

    Args:
        found_names: список имен/фамилий, извлеченных из транскрипта
        allowed_names: официальный список допустимых ФИО

    Returns:
        Результаты сопоставления всех найденных имён.
    """
    if not found_names.names:
        return ValidationResult(results=[])

    # Материализация списка позволяет использовать данные Google Sheets в тексте запроса модели.
    allowed_names_list = [name.strip() for name in allowed_names if name.strip()]

    # Точное совпадение фамилии и имени надёжнее модели и не зависит от наличия отчества.
    official_names = _build_official_name_index(allowed_names_list)
    deterministic_results: dict[int, ValidatedStudent] = {}
    unmatched_names = []
    unmatched_indexes = []
    for index, found_name in enumerate(found_names.names):
        matching_name = _match_surname_and_given_name(found_name.name, official_names)
        if matching_name is None:
            unmatched_names.append(found_name)
            unmatched_indexes.append(index)
            continue
        deterministic_results[index] = ValidatedStudent(
            raw_name=found_name.name,
            status="valid",
            matched_name=matching_name,
            confidence=100,
            comment="Точное совпадение фамилии и имени; отчество не учитывается",
            minute=found_name.minute_start,
        )

    if not unmatched_names:
        return ValidationResult(
            results=[deterministic_results[index] for index in range(len(found_names.names))]
        )

    prompt = f"""
Ты валидируешь найденные из транскрипта ФИО студентов по официальному списку.

Тебе даны:
1. Список найденных имен из распознавания речи и минута, когда студент говорил
2. Список допустимых ФИО студентов

Задача:
Для каждого элемента из списка найденных имен определи:
- valid: если это уверенно соответствует одному студенту из официального списка
- uncertain: если есть вероятное совпадение, но есть сомнения
- invalid: если соответствия нет

Правила:
- Сопоставляй только с именами из официального списка.
- Учитывай возможные ошибки распознавания речи, опечатки, перестановку слов, падежи, неполные и искаженные формы.
- Если найдено несколько похожих кандидатов и нельзя уверенно выбрать одного, ставь uncertain.
- Если имя явно не соответствует никому из официального списка, ставь invalid.
- Для valid и uncertain заполняй matched_name.
- Для invalid matched_name должен быть null.
- confidence:
  - 90-100 для уверенного совпадения
  - 60-89 для uncertain
  - 0-59 для invalid
- comment должен быть коротким и конкретным.
- Верни результат для каждого найденного имени.
- Не придумывай новых людей вне официального списка.

Найденные имена:
{[f'{v.name} ({v.minute_start})' for v in unmatched_names]}

Официальный список студентов:
{allowed_names_list}
"""

    structured_llm = llm.with_structured_output(ValidationResult)
    model_result = structured_llm.invoke(prompt)
    for index, validated_student in zip(unmatched_indexes, model_result.results, strict=True):
        deterministic_results[index] = validated_student

    return ValidationResult(
        results=[deterministic_results[index] for index in range(len(found_names.names))]
    )


def _build_official_name_index(allowed_names: Iterable[str]) -> dict[tuple[str, str], list[str]]:
    """Создать индекс официальных ФИО по паре «фамилия, имя».

    Args:
        allowed_names: ФИО из официального списка студентов.

    Returns:
        Сопоставление нормализованной пары с оригинальными ФИО.
    """
    # Список значений сохраняет неоднозначные пары, которые нельзя подтверждать автоматически.
    index: dict[tuple[str, str], list[str]] = {}
    for allowed_name in allowed_names:
        tokens = _name_tokens(allowed_name)
        if len(tokens) < 2:
            continue
        index.setdefault((tokens[0], tokens[1]), []).append(allowed_name)
    return index


def _match_surname_and_given_name(
    found_name: str,
    official_names: dict[tuple[str, str], list[str]],
) -> str | None:
    """Найти однозначное официальное ФИО по фамилии и имени.

    Args:
        found_name: ФИО, извлечённое из транскрипта.
        official_names: индекс официального списка студентов.

    Returns:
        Полное официальное ФИО или ``None`` для отсутствующего либо неоднозначного совпадения.
    """
    # Отчество намеренно не участвует в ключе, поскольку в речи оно обычно отсутствует.
    tokens = _name_tokens(found_name)
    if len(tokens) < 2:
        return None
    matches = official_names.get((tokens[0], tokens[1]), [])
    return matches[0] if len(matches) == 1 else None


def _name_tokens(name: str) -> list[str]:
    """Нормализовать ФИО для устойчивого сравнения.

    Args:
        name: исходное ФИО.

    Returns:
        Список буквенных токенов в нижнем регистре.
    """
    # Нормализация убирает различия «ё/е» и комбинируемые акценты из таблицы.
    normalized = unicodedata.normalize("NFKD", name.lower().replace("ё", "е"))
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.findall(r"[a-zа-я]+", normalized)
