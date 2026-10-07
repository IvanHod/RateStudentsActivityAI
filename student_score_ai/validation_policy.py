"""Единая политика версий, ревью и начисления баллов."""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ValidationPolicy:
    """Описать правила одного выпуска алгоритма валидации.

    Attributes:
        version: номер формата результата валидации.
        score_confidence_threshold: строгий нижний порог confidence для начисления.
        review_only: признак запрета на автоматическую запись в Google Sheets.
    """

    version: int
    score_confidence_threshold: int
    review_only: bool


# Одна политика нужна всем точкам входа, чтобы номер артефакта и правило начисления не расходились.
VALIDATION_POLICY: Final = ValidationPolicy(
    version=6,
    score_confidence_threshold=30,
    review_only=True,
)
