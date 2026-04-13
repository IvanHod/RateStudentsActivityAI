import os
import re
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from pydantic import BaseModel
from langchain_openai import ChatOpenAI


class NameList(BaseModel):
    names: list[str]


class FoundedName(BaseModel):
    name: str
    minute_start: int


class NameListWithTime(BaseModel):
    names: list[FoundedName]


llm = ChatOpenAI(
    model="gemma-4-26B-A4B-it",
    temperature=0.1,
    api_key=os.environ["LITELLM_API_KEY"],          # ключ LiteLLM proxy
    base_url=os.environ["LITELLM_BASE_URL"],        # например http://my-host:4000/v1
)


def extract_student_names(path_transcript: Iterable[Path]) -> NameListWithTime:
    prompt = """
Ты получаешь транскрипт лекции.
Нужно извлечь все упомянутые полные имена студентов в формате:
["Фамилия Имя", "Фамилия Имя", ...]

Правила:
- Верни только реальные ФИО людей.
- Если имя неполное, не включай его.
- Ответ верни строго JSON.

Транскрипт:
{transcript}
"""

    names_list = []
    for path in tqdm(sorted(path_transcript, key=lambda p: int(p.stem[-4:]))):
        with open(path, "r", encoding="utf-8") as file:
            transcript = file.read()

            structured_llm = llm.with_structured_output(NameList, strict=True)
            result = structured_llm.invoke(prompt.format(transcript=transcript))

            minute_start = int(re.search(r'_(\d+)\.', path.name).group(1))
            for name in result.names:
                names_list.append({'name': name, 'minute_start': minute_start})

    return NameListWithTime.model_validate({"names": names_list})