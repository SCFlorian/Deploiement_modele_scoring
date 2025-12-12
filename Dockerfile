FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY . .

EXPOSE 7860

CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
