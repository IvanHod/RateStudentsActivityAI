import os
import re
import time
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from langchain_openai import ChatOpenAI

from student_score_ai.extract_names import FoundedName, NameListWithTime

# Explicit retry settings make a stalled model request observable in console output.
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_ATTEMPTS = 4
LLM_RETRY_DELAYS_SECONDS = (1, 2, 4)

# HTTP hooks expose the actual endpoint and response timing without logging credentials.
def _log_llm_request(request: httpx.Request) -> None:
    """Записать начало HTTP-запроса к языковой модели.

    Args:
        request: исходящий HTTP-запрос без необходимости читать его тело.

    Returns:
        None.
    """
    # Монотонные часы позволяют корректно измерить запрос, даже если меняется системное время.
    request.extensions["student_score_started_at"] = time.monotonic()
    print(f"Шаг 4/5: LiteLLM HTTP → {request.method} {request.url}")


def _log_llm_response(response: httpx.Response) -> None:
    """Записать код и длительность полученного ответа LiteLLM.

    Args:
        response: HTTP-ответ от прокси LiteLLM.

    Returns:
        None.
    """
    # Время известно уже после заголовков и помогает отличить ответ proxy от таймаута сети.
    started_at = response.request.extensions.get("student_score_started_at")
    elapsed = time.monotonic() - started_at if isinstance(started_at, float) else 0.0
    print(
        f"Шаг 4/5: LiteLLM HTTP ← {response.status_code} "
        f"за {elapsed:.1f} с: {response.request.url}"
    )


# A dedicated client is needed so each request can report its real URL and status.
llm_http_client = httpx.Client(
    timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS),
    event_hooks={"request": [_log_llm_request], "response": [_log_llm_response]},
)

llm = ChatOpenAI(
    model="gemma-4-26B-A4B-it",
    temperature=0.1,
    max_retries=0,
    timeout=LLM_TIMEOUT_SECONDS,
    api_key=os.environ["LITELLM_API_KEY"],          # ключ LiteLLM proxy
    base_url=os.environ["LITELLM_BASE_URL"],        # например http://my-host:4000/v1
    http_client=llm_http_client,
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

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> Any:
        """Привести дробную уверенность модели к целому проценту.

        Args:
            value: значение уверенности из JSON-ответа модели.

        Returns:
            Целый процент для числового значения либо исходное значение для Pydantic.
        """
        # Некоторые модели используют вероятность 0..1 вместо явно запрошенных процентов 0..100.
        if isinstance(value, bool) or not isinstance(value, Real):
            return value
        if 0 <= value <= 1:
            return round(value * 100)
        return round(value)


class ValidationResult(BaseModel):
    results: list[ValidatedStudent]


class CandidateMatch(BaseModel):
    """Кандидат из реестра, ранжированный по сходству с фрагментом.

    Attributes:
        name: полное официальное ФИО студента.
        similarity: нормализованная оценка сходства от 0 до 100.
    """

    name: str
    similarity: int = Field(ge=0, le=100)


def validate_student_names_batch_with_gpt(
    found_names: NameListWithTime,
    allowed_names: Iterable[str],
    transcript_paths: Iterable[Path],
) -> ValidationResult:
    """Проверить фрагменты по кандидатам из реестра и их контексту.

    Args:
        found_names: ФИО и минуты, извлечённые из транскрипта.
        allowed_names: полный официальный список ФИО студентов.
        transcript_paths: минутные фрагменты транскрипта для проверки контекстом.

    Returns:
        Результаты в том же порядке, что и ``found_names``.
    """
    # Полный список не передаётся модели: ранжирование снижает число случайных сопоставлений.
    return validate_student_names_with_gpt(
        found_names=found_names,
        allowed_names=allowed_names,
        transcript_paths=transcript_paths,
    )


def _recheck_ambiguous_batch_results(
    batch_result: ValidationResult,
    found_names: NameListWithTime,
    allowed_names: list[str],
    transcript_paths: Iterable[Path],
) -> ValidationResult:
    """Уточнить сомнительные пакетные решения по тексту соответствующей минуты.

    Args:
        batch_result: результаты первоначальной проверки полным реестром.
        found_names: исходные ФИО и минуты из конвейера.
        allowed_names: полный официальный список студентов.
        transcript_paths: минутные фрагменты транскрипта занятия.

    Returns:
        Итоговые результаты с контекстной проверкой всех ``invalid`` и ``uncertain``.
    """
    # Контекстная проверка всех неточных фрагментов не даёт пакетной модели подтвердить случайное сходство.
    transcript_contexts = _build_transcript_contexts(transcript_paths)
    rechecked_results: list[ValidatedStudent] = []
    official_names = _build_official_name_index(allowed_names)
    non_exact_count = sum(
        _match_surname_and_given_name(source.name, official_names) is None
        for source in found_names.names
    )
    current_non_exact = 0
    for source, _result in zip(found_names.names, batch_result.results, strict=True):
        matching_name = _match_surname_and_given_name(source.name, official_names)
        if matching_name is not None:
            rechecked_results.append(
                ValidatedStudent(
                    raw_name=source.name,
                    status="valid",
                    matched_name=matching_name,
                    confidence=100,
                    comment="Точное совпадение фамилии и имени; отчество не учитывается",
                    minute=source.minute_start,
                )
            )
            continue
        current_non_exact += 1
        print(
            f"Шаг 4/5: контекстное уточнение {current_non_exact}/{non_exact_count} "
            f"— {source.name}"
        )
        candidates = _candidate_names(source.name, allowed_names)
        context = _context_for_source(
            source=source,
            transcript=transcript_contexts.get(source.minute_start, ""),
        )
        rechecked_results.append(
            _validate_name_with_context(
                found_name=source.name,
                minute=source.minute_start,
                context=context,
                candidates=candidates,
            )
        )
    return ValidationResult(results=rechecked_results)


def _invoke_batch_validation_llm(prompt: str) -> ValidationResult:
    """Запросить пакетную валидацию с контролируемыми повторами.

    Args:
        prompt: инструкция и данные для пакетной проверки.

    Returns:
        Структурированный ответ модели до привязки к исходным данным.

    Raises:
        RuntimeError: если все попытки запроса закончились ошибкой.
    """
    # Ручные повторы сохраняют наблюдаемость сетевых ошибок, как и в построчной стратегии.
    last_error: Exception | None = None
    structured_llm = llm.with_structured_output(ValidationResult)
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        print(
            f"Шаг 4/5: пакетный запрос LiteLLM, попытка {attempt}/{LLM_MAX_ATTEMPTS} "
            f"(таймаут {LLM_TIMEOUT_SECONDS} с)"
        )
        try:
            return structured_llm.invoke(prompt)
        except Exception as error:
            last_error = error
            if attempt == LLM_MAX_ATTEMPTS:
                break
            retry_delay = LLM_RETRY_DELAYS_SECONDS[attempt - 1]
            print(
                "Шаг 4/5: пакетный LiteLLM не ответил: "
                f"{_format_exception_chain(error)}; повтор через {retry_delay} с"
            )
            time.sleep(retry_delay)

    raise RuntimeError(
        "LiteLLM не ответил при пакетной проверке ФИО после "
        f"{LLM_MAX_ATTEMPTS} попыток по {LLM_TIMEOUT_SECONDS} секунд"
    ) from last_error


def _reconcile_batch_validation_result(
    model_result: ValidationResult,
    found_names: NameListWithTime,
    allowed_names: list[str],
) -> ValidationResult:
    """Привязать ответ модели к исходным ФИО и проверить его границы.

    Args:
        model_result: ответ модели в предполагаемом порядке.
        found_names: исходные ФИО и минуты из конвейера.
        allowed_names: полный список допустимых официальных ФИО.

    Returns:
        Результаты с неизменяемыми ``raw_name`` и ``minute`` из конвейера.

    Raises:
        ValueError: если число ответов не совпадает или выбран недопустимый студент.
    """
    # Позиционное сопоставление не даёт модели случайно превратить минуту в часть имени.
    if len(model_result.results) != len(found_names.names):
        raise ValueError(
            "Пакетная валидация вернула "
            f"{len(model_result.results)} результатов вместо {len(found_names.names)}"
        )

    reconciled_results: list[ValidatedStudent] = []
    for source, result in zip(found_names.names, model_result.results, strict=True):
        if result.matched_name is not None and result.matched_name not in allowed_names:
            raise ValueError(
                f"Пакетная валидация вернула ФИО вне официального списка: "
                f"{result.matched_name}"
            )
        reconciled_results.append(
            result.model_copy(
                update={"raw_name": source.name, "minute": source.minute_start}
            )
        )
    return ValidationResult(results=reconciled_results)


def validate_student_names_with_gpt(
    found_names: NameListWithTime,
    allowed_names: Iterable[str],
    transcript_paths: Iterable[Path],
) -> ValidationResult:
    """
    Валидирует найденные ФИО по официальному списку через GPT.

    Args:
        found_names: список имен/фамилий, извлеченных из транскрипта
        allowed_names: официальный список допустимых ФИО
        transcript_paths: пути к минутным фрагментам транскрипта

    Returns:
        Результаты сопоставления всех найденных имён.
    """
    if not found_names.names:
        return ValidationResult(results=[])

    # Материализация списка позволяет использовать данные Google Sheets в тексте запроса модели.
    allowed_names_list = [name.strip() for name in allowed_names if name.strip()]
    transcript_contexts = _build_transcript_contexts(transcript_paths)

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

    # Контекст и короткий список кандидатов снижают риск случайного выбора из всего потока.
    unmatched_count = len(unmatched_names)
    for position, (index, found_name) in enumerate(
        zip(unmatched_indexes, unmatched_names, strict=True),
        start=1,
    ):
        print(
            f"Шаг 4/5: контекстная проверка {position}/{unmatched_count} "
            f"— {found_name.name}"
        )
        candidates = _candidate_names(found_name.name, allowed_names_list)
        context = _context_for_source(
            source=found_name,
            transcript=transcript_contexts.get(found_name.minute_start, ""),
        )
        deterministic_results[index] = _validate_name_with_context(
            found_name.name,
            found_name.minute_start,
            context,
            candidates,
        )

    return ValidationResult(
        results=[deterministic_results[index] for index in range(len(found_names.names))]
    )


def _validate_name_with_context(
    found_name: str,
    minute: int,
    context: str,
    candidates: list[CandidateMatch],
) -> ValidatedStudent:
    """Проверить одно неочевидное ФИО по контексту и похожим студентам.

    Args:
        found_name: ФИО, извлечённое из транскрипта.
        minute: минута исходной записи.
        context: фрагмент транскрипта вокруг упоминания.
        candidates: наиболее похожие официальные ФИО и оценки их сходства.

    Returns:
        Результат контролируемого сопоставления.
    """
    # Модель видит только реальные кандидаты, поэтому не может подставить студента вне списка.
    prompt = f"""
Ты проверяешь одно предположительное ФИО студента в транскрипте занятия.

Найденное ФИО: {found_name}
Минута записи: {minute}
Фрагмент транскрипта:
{context}

Похожие студенты из официального списка:
{[candidate.model_dump() for candidate in candidates]}

Правила:
- matched_name может быть только одним из приведённых официальных ФИО.
- valid ставь только если контекст, фрагмент и один кандидат согласуются. Высокая
  similarity сама по себе недостаточна.
- Если контекст говорит, что фамилия или имя названы неточно, проверь фонетически близкого кандидата.
- Если совпадает только имя, только фамилия, есть несколько близких кандидатов или
  контекст неясен, ставь uncertain.
- Если подходящего кандидата нет, ставь invalid и matched_name=null.
- Не выдумывай ФИО и не используй кандидатов вне списка.
- Ответь только одним JSON-объектом без Markdown и без пояснений вокруг него.
- JSON обязан содержать поля raw_name, status, matched_name, confidence, comment и minute.
- confidence указывай целым числом процентов от 0 до 100, например 80, а не 0.8.
"""
    validated = _invoke_validation_llm(prompt, found_name)

    # Исходные имя и минута принадлежат конвейеру, а не ответу модели.
    validated = validated.model_copy(update={"raw_name": found_name, "minute": minute})
    candidate_names = {candidate.name for candidate in candidates}
    if validated.matched_name not in candidate_names:
        return ValidatedStudent(
            raw_name=found_name,
            status="invalid",
            matched_name=None,
            confidence=0,
            comment="Модель вернула ФИО вне списка кандидатов",
            minute=minute,
        )
    return validated


def _invoke_validation_llm(prompt: str, found_name: str) -> ValidatedStudent:
    """Вызвать модель с видимыми в консоли повторными попытками.

    Args:
        prompt: подготовленная инструкция для модели.
        found_name: ФИО, используемое только в сообщениях журнала.

    Returns:
        Валидированный структурированный ответ модели.

    Raises:
        RuntimeError: если модель не ответила за все разрешённые попытки.
    """
    # Повторы выполняются здесь, а не в SDK, чтобы оператор видел каждую попытку и паузу.
    last_error: Exception | None = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        print(
            f"Шаг 4/5: запрос LiteLLM для «{found_name}», "
            f"попытка {attempt}/{LLM_MAX_ATTEMPTS} "
            f"(таймаут {LLM_TIMEOUT_SECONDS} с)"
        )
        try:
            response = llm.invoke(prompt)
            return _parse_validated_student_response(response.content)
        except Exception as error:
            last_error = error
            if attempt == LLM_MAX_ATTEMPTS:
                break

            retry_delay = LLM_RETRY_DELAYS_SECONDS[attempt - 1]
            print(
                f"Шаг 4/5: LiteLLM не ответил для «{found_name}»: "
                f"{_format_exception_chain(error)}; повтор через {retry_delay} с"
            )
            time.sleep(retry_delay)

    # Последняя ошибка остаётся причиной исключения, чтобы traceback содержал детали SDK.
    raise RuntimeError(
        f"LiteLLM не ответил при проверке ФИО «{found_name}» после "
        f"{LLM_MAX_ATTEMPTS} попыток по {LLM_TIMEOUT_SECONDS} секунд"
    ) from last_error


def _parse_validated_student_response(content: str | list[str | dict[str, Any]]) -> ValidatedStudent:
    """Преобразовать текстовый JSON-ответ модели в проверенный результат.

    Args:
        content: содержимое ответа LangChain в строковом либо блочном виде.

    Returns:
        Результат проверки ФИО с типами, проверенными Pydantic.

    Raises:
        ValueError: если ответ не содержит текстового JSON-объекта.
        pydantic.ValidationError: если JSON не соответствует ожидаемой схеме.
    """
    # Gemma отвечает текстом, а обработка блочного формата сохраняет совместимость с API OpenAI.
    response_text = _response_content_to_text(content)
    json_text = _strip_json_markdown_fence(response_text)
    return ValidatedStudent.model_validate_json(json_text)


def _response_content_to_text(content: str | list[str | dict[str, Any]]) -> str:
    """Извлечь текст из строкового или блочного содержимого ответа модели.

    Args:
        content: содержимое сообщения, возвращённого LangChain.

    Returns:
        Непустой текст, который можно передать JSON-парсеру.

    Raises:
        ValueError: если ответ не содержит текстовых блоков.
    """
    # OpenAI-совместимые серверы могут вернуть текст как строку или список блоков.
    if isinstance(content, str):
        response_text = content.strip()
    else:
        text_blocks = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
        ]
        response_text = "".join(text_blocks).strip()
    if not response_text:
        raise ValueError("LiteLLM вернул ответ без текстового содержимого")
    return response_text


def _strip_json_markdown_fence(response_text: str) -> str:
    """Убрать необязательное Markdown-ограждение вокруг JSON модели.

    Args:
        response_text: текст, предположительно содержащий JSON-объект.

    Returns:
        Текст JSON без внешнего Markdown-ограждения.
    """
    # Модель может нарушить инструкцию и обернуть корректный JSON в блок ```json.
    if not response_text.startswith("```"):
        return response_text
    lines = response_text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return response_text


def _format_exception_chain(error: Exception) -> str:
    """Сформировать краткую безопасную цепочку причин ошибки HTTP-клиента.

    Args:
        error: верхнеуровневое исключение, возвращённое LangChain или OpenAI SDK.

    Returns:
        Типы и непустые описания исключений от внешнего к внутреннему.
    """
    # Цепочка причин показывает фазу сбоя сети, не печатая заголовки с API-ключом.
    descriptions: list[str] = []
    current_error: BaseException | None = error
    while current_error is not None:
        detail = str(current_error).strip()
        descriptions.append(
            f"{type(current_error).__name__}{f': {detail}' if detail else ''}"
        )
        cause = current_error.__cause__ or current_error.__context__
        current_error = cause if isinstance(cause, BaseException) else None
    return " ← ".join(descriptions)


def _candidate_names(
    found_name: str,
    allowed_names: Iterable[str],
    limit: int = 6,
) -> list[CandidateMatch]:
    """Выбрать ранжированный список официальных ФИО для искажённого фрагмента.

    Args:
        found_name: извлечённое ФИО.
        allowed_names: официальный список студентов.
        limit: максимальное число кандидатов.

    Returns:
        Наиболее похожие полные ФИО с оценкой сходства.
    """
    # Сравнение склеенных и фонетических форм сохраняет кандидата после типичных ошибок Whisper.
    scored_candidates: list[tuple[float, str]] = []
    for allowed_name in allowed_names:
        scored_candidates.append((_candidate_similarity(found_name, allowed_name), allowed_name))
    return [
        CandidateMatch(name=name, similarity=round(score * 100))
        for score, name in sorted(scored_candidates, reverse=True)[:limit]
    ]


def _candidate_similarity(fragment: str, official_name: str) -> float:
    """Оценить строковое и фонетическое сходство фрагмента с ФИО из реестра.

    Args:
        fragment: сырая строка из транскрипта.
        official_name: полное официальное ФИО студента.

    Returns:
        Оценка сходства в диапазоне от 0 до 1.
    """
    # Склеенная форма важна для ASR, который часто не ставит границу между фамилией и именем.
    fragment_tokens = _name_tokens(fragment)
    official_tokens = _name_tokens(official_name)
    if len(official_tokens) < 2:
        return 0.0
    fragment_joined = "".join(fragment_tokens)
    official_joined = "".join(official_tokens[:2])
    direct_scores = [
        SequenceMatcher(None, fragment_joined, official_joined).ratio(),
        SequenceMatcher(None, _phonetic_key(fragment_joined), _phonetic_key(official_joined)).ratio(),
    ]
    if len(fragment_tokens) >= 2:
        direct_scores.extend(
            [
                0.7 * SequenceMatcher(None, fragment_tokens[0], official_tokens[0]).ratio()
                + 0.3 * SequenceMatcher(None, fragment_tokens[1], official_tokens[1]).ratio(),
                0.7 * SequenceMatcher(
                    None,
                    _phonetic_key(fragment_tokens[0]),
                    _phonetic_key(official_tokens[0]),
                ).ratio()
                + 0.3 * SequenceMatcher(
                    None,
                    _phonetic_key(fragment_tokens[1]),
                    _phonetic_key(official_tokens[1]),
                ).ratio(),
            ]
        )
    return max(direct_scores)


def _phonetic_key(text: str) -> str:
    """Свести распространённые русские фонетические варианты к одному ключу.

    Args:
        text: нормализованная последовательность русских букв.

    Returns:
        Упрощённое фонетическое представление строки.
    """
    # Подстановки покрывают частые ошибки ASR, но решение всё равно принимает модель по контексту.
    normalized = text
    for source, target in (
        ("тс", "ц"),
        ("дс", "ц"),
        ("тч", "ч"),
        ("сч", "щ"),
        ("ж", "ш"),
        ("з", "с"),
        ("б", "п"),
        ("в", "ф"),
        ("г", "к"),
        ("д", "т"),
        ("ъ", ""),
        ("ь", ""),
        ("й", "и"),
        ("ы", "и"),
        ("я", "а"),
        ("ю", "у"),
        ("е", "и"),
        ("о", "а"),
    ):
        normalized = normalized.replace(source, target)
    return re.sub(r"(.)\\1+", r"\\1", normalized)


def _build_transcript_contexts(transcript_paths: Iterable[Path]) -> dict[int, str]:
    """Прочитать минутные фрагменты транскрипта по их номеру.

    Args:
        transcript_paths: пути к текстовым файлам чанков.

    Returns:
        Сопоставление номера минуты с текстом соответствующего чанка.
    """
    # Индекс по минуте позволяет связать ФИО с оригинальной фразой преподавателя.
    contexts: dict[int, str] = {}
    for path in transcript_paths:
        try:
            minute = int(path.stem.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        contexts[minute] = path.read_text(encoding="utf-8")
    return contexts


def _context_for_source(source: FoundedName, transcript: str) -> str:
    """Выбрать сохранённый контекст извлечения либо восстановить его из чанка.

    Args:
        source: фрагмент, переданный извлекателем.
        transcript: полный текст минутного чанка.

    Returns:
        Контекст, пригодный для проверки кандидатов.
    """
    # Сохранённый контекст точнее, когда фрагмент искажён настолько, что не находится подстрокой.
    if source.context.strip():
        return source.context
    return _transcript_context(transcript, source.name)


def _transcript_context(transcript: str, found_name: str, radius: int = 400) -> str:
    """Получить небольшой фрагмент транскрипта вокруг найденного ФИО.

    Args:
        transcript: полный текст минутного чанка.
        found_name: ФИО, которое нужно найти в тексте.
        radius: число символов с каждой стороны упоминания.

    Returns:
        Контекст с упоминанием либо начало чанка, если точной подстроки нет.
    """
    # Короткий контекст уменьшает стоимость запроса и сохраняет реплику, важную для решения.
    position = transcript.lower().find(found_name.lower())
    if position < 0:
        return transcript[: radius * 2]
    return transcript[max(0, position - radius): position + len(found_name) + radius]


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
