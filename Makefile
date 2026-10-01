SHELL := /bin/bash

PYTHON := PYTHONPATH=. uv run python
OUTPUT_DIR := output
VALIDATION_VERSION := $(shell PYTHONPATH=. python3 -c 'from student_score_ai.validation_policy import VALIDATION_POLICY; print(VALIDATION_POLICY.version)')
VALIDATION_FILE := students_validated_v$(VALIDATION_VERSION).json
POSITIONAL_WEEK_DATE := $(filter-out run_week,$(MAKECMDGOALS))
WEEK_DATE := $(strip $(or $(DATE),$(POSITIONAL_WEEK_DATE)))

ifneq (,$(filter run_week,$(MAKECMDGOALS)))
ifneq (,$(POSITIONAL_WEEK_DATE))
.PHONY: $(POSITIONAL_WEEK_DATE)
$(POSITIONAL_WEEK_DATE):
	@:
endif
endif

.DEFAULT_GOAL := help
.PHONY: help run_all run_week run_lecture score_preview_all score_compare_sheet_all score_rebuild_preview_all score_rebuild_all score_apply_all

help:
	@printf '%s\n' \
	  'make run_all' \
	  'make run_week DATE=07.10.2026' \
	  'make run_week 07.10.2026' \
	  'make run_lecture LECTURE="Лекция1_2компилятор_поток1_14.09.2026.mp4"' \
	  'make score_preview_all' \
	  'make score_compare_sheet_all' \
	  'make score_rebuild_preview_all' \
	  'make score_rebuild_all CONFIRM=REBUILD' \
	  'make score_apply_all CONFIRM=APPLY'

run_all:
	$(PYTHON) -m student_score_ai.main

run_week:
	@test -n "$(WEEK_DATE)" || { echo 'Укажите DATE=DD.MM.YYYY' >&2; exit 2; }
	$(PYTHON) -m student_score_ai.main --week "$(WEEK_DATE)"

run_lecture:
	@test -n "$(LECTURE)" || { echo 'Укажите LECTURE="точное имя файла"' >&2; exit 2; }
	$(PYTHON) -m student_score_ai.main --lecture "$(LECTURE)"

score_preview_all:
	@set -euo pipefail; count=0; \
	while IFS= read -r -d '' validation_file; do \
	  work_dir="$$(dirname "$$validation_file")"; \
	  $(PYTHON) -m student_score_ai.review_scores "$$work_dir"; \
	  count=$$((count + 1)); \
	done < <(find "$(OUTPUT_DIR)" -type f -name "$(VALIDATION_FILE)" -print0); \
	if [ "$$count" -eq 0 ]; then \
	  echo 'Не найдены результаты валидации v$(VALIDATION_VERSION)' >&2; exit 1; \
	fi

score_compare_sheet_all:
	@set -euo pipefail; count=0; \
	while IFS= read -r -d '' validation_file; do \
	  work_dir="$$(dirname "$$validation_file")"; \
	  $(PYTHON) -m student_score_ai.review_scores "$$work_dir" --compare-sheet; \
	  count=$$((count + 1)); \
	done < <(find "$(OUTPUT_DIR)" -type f -name "$(VALIDATION_FILE)" -print0); \
	if [ "$$count" -eq 0 ]; then \
	  echo 'Не найдены результаты валидации v$(VALIDATION_VERSION)' >&2; exit 1; \
	fi

score_rebuild_preview_all:
	@set -euo pipefail; count=0; \
	while IFS= read -r -d '' validation_file; do \
	  work_dir="$$(dirname "$$validation_file")"; \
	  $(PYTHON) -m student_score_ai.review_scores "$$work_dir" --rebuild-preview; \
	  count=$$((count + 1)); \
	done < <(find "$(OUTPUT_DIR)" -type f -name "$(VALIDATION_FILE)" -print0); \
	if [ "$$count" -eq 0 ]; then \
	  echo 'Не найдены результаты валидации v$(VALIDATION_VERSION)' >&2; exit 1; \
	fi

score_rebuild_all:
	@test "$(CONFIRM)" = 'REBUILD' || { echo 'Для полного восстановления после очистки таблицы запустите: make score_rebuild_all CONFIRM=REBUILD' >&2; exit 2; }
	@set -euo pipefail; \
	while IFS= read -r -d '' completion_file; do \
	  work_dir="$$(dirname "$$completion_file")"; \
	  test -f "$$work_dir/$(VALIDATION_FILE)" || { echo "Нет $$work_dir/$(VALIDATION_FILE); сначала запустите make run_all" >&2; exit 1; }; \
	done < <(find "$(OUTPUT_DIR)" -type f -name completed.json -print0); \
	count=0; \
	while IFS= read -r -d '' validation_file; do \
	  work_dir="$$(dirname "$$validation_file")"; \
	  $(PYTHON) -m student_score_ai.review_scores "$$work_dir" --rebuild; \
	  count=$$((count + 1)); \
	done < <(find "$(OUTPUT_DIR)" -type f -name "$(VALIDATION_FILE)" -print0); \
	if [ "$$count" -eq 0 ]; then \
	  echo 'Не найдены результаты валидации v$(VALIDATION_VERSION)' >&2; exit 1; \
	fi

score_apply_all:
	@test "$(CONFIRM)" = 'APPLY' || { echo 'Для записи в Google Sheets запустите: make score_apply_all CONFIRM=APPLY' >&2; exit 2; }
	@set -euo pipefail; count=0; \
	while IFS= read -r -d '' validation_file; do \
	  work_dir="$$(dirname "$$validation_file")"; \
	  $(PYTHON) -m student_score_ai.review_scores "$$work_dir" --apply; \
	  count=$$((count + 1)); \
	done < <(find "$(OUTPUT_DIR)" -type f -name "$(VALIDATION_FILE)" -print0); \
	if [ "$$count" -eq 0 ]; then \
	  echo 'Не найдены результаты валидации v$(VALIDATION_VERSION)' >&2; exit 1; \
	fi
