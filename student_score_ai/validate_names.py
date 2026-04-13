import os
from pathlib import Path
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
    path_names: Path,
) -> ValidationResult:
    """
    Валидирует найденные ФИО по официальному списку через GPT.

    Args:
        found_names: список имен/фамилий, извлеченных из транскрипта
        path_names: путь к официальному списку допустимых ФИО

    Returns:
        list[ValidatedStudent]
    """
    if not found_names.names:
        return found_names

    with open(path_names, "r", encoding="utf-8") as f:
        allowed_names = f.read().split('\n')

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
{[f'{v.name} ({v.minute_start})' for v in found_names.names]}

Официальный список студентов:
{allowed_names}
"""

    structured_llm = llm.with_structured_output(ValidationResult)
    result = structured_llm.invoke(prompt)

    return result