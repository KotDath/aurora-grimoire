# Aurora Grimoire

CLI + MCP сервер для RAG по документации Aurora OS.

Проект покрывает полный цикл:
- выкачивание HTML-документации;
- нормализация в Markdown;
- чанкование;
- генерация эмбеддингов;
- загрузка в Qdrant;
- поиск с источниками (`cli search_docs`);
- доступ к поиску через MCP tool `search_docs`.

## Возможности

- Полный offline-friendly RAG pipeline через CLI.
- Hybrid retrieval: dense (Qdrant) + BM25.
- Фильтрация по версии документации (`--doc-version`).
- Порог "достаточности знания" (`--knowledge-threshold`).
- Перенос артефактов между машинами через bundle.
- MCP сервер (`stdio` и HTTP transport).

## Требования

- Rust/Cargo (для установки CLI через `cargo install --path .`).
- Для `rag embed`: Ollama с моделью эмбеддингов.
- Для `rag deploy` и поиска: Qdrant.

Если нужен локальный стенд:
- `rag dev up` поднимет Docker stack (Qdrant + Ollama).

## Установка

Установить CLI-бинарь:

```bash
cargo install --path .
```

Проверить установку:

```bash
aurora-grimoire --version
```

Если команда не находится, добавьте Cargo bin в `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Конфиг обязателен

Все команды, кроме `rag config init`, требуют существующий `config.toml`.

Создать конфиг:

```bash
aurora-grimoire rag config init
```

С перезаписью:

```bash
aurora-grimoire rag config init --force
```

Кастомный путь:

```bash
AURORA_GRIMOIRE_CONFIG=/path/to/config.toml aurora-grimoire rag config init
```

Шаблон конфига: [`config.example.toml`](config.example.toml)

## Быстрый старт (end-to-end)

1) Инициализировать конфиг:

```bash
aurora-grimoire rag config init
```

2) Поднять локальный stack:

```bash
aurora-grimoire rag dev up --build --gpu --verbose
```

3) Выкачать и подготовить данные:

```bash
aurora-grimoire rag fetch-web -v
aurora-grimoire rag struct -v
aurora-grimoire rag chunk -v
```

4) Сгенерировать эмбеддинги и загрузить в Qdrant:

```bash
aurora-grimoire rag embed -v
aurora-grimoire rag deploy -v --recreate
```

5) Проверить поиск:

```bash
aurora-grimoire cli search_docs "как собрать проект через mb2" --doc-version 5.2.0 --json
```

## Перенос артефактов другому пользователю

`rag bundle create` всегда создает `dual` bundle (`chunks + vectors`).

Создать bundle:

```bash
aurora-grimoire rag bundle create --out ~/.aurora-grimoire/bundles/aurora-dual.tar.zst
```

Проверить bundle:

```bash
aurora-grimoire rag bundle inspect --file ~/.aurora-grimoire/bundles/aurora-dual.tar.zst
```

На другой машине:

```bash
aurora-grimoire rag config init
aurora-grimoire rag bundle extract --file /path/aurora-dual.tar.zst
aurora-grimoire rag deploy --input ~/.aurora-grimoire/vectors_data --recreate -v
```

## MCP

Запуск MCP сервера:

```bash
aurora-grimoire mcp start --stdio
```

или HTTP:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
```

Smoke-проверка MCP:

```bash
aurora-grimoire mcp smoke --doc-version 5.2.0 --query "как собрать проект через mb2"
```

MCP предоставляет один tool:
- `search_docs`

## Подключение к разным агентам

Убедитесь, что `aurora-grimoire` установлен и доступен в `PATH`.
Если нужен абсолютный путь к бинарю:

```bash
AURORA_GRIMOIRE_BIN="$(command -v aurora-grimoire)"
```

Используйте этот абсолютный путь в примерах ниже.

### Claude Code

Stdio transport:

```bash
claude mcp add aurora-grimoire -- $AURORA_GRIMOIRE_BIN mcp start --stdio
```

HTTP transport:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
claude mcp add --transport http aurora-grimoire-http http://127.0.0.1:8080/mcp
```

### Codex

Stdio transport:

```bash
codex mcp add aurora-grimoire -- $AURORA_GRIMOIRE_BIN mcp start --stdio
```

HTTP transport:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
codex mcp add aurora-grimoire-http --url http://127.0.0.1:8080/mcp
```

Проверка:

```bash
codex mcp list
```

### OpenCode

OpenCode добавляет MCP сервер через интерактивный wizard:

```bash
opencode mcp add
```

Для локального режима укажите:
- `name`: `aurora-grimoire`
- `transport/type`: `local`/`stdio`
- `command`: `$AURORA_GRIMOIRE_BIN`
- `args`: `mcp start --stdio`

Для удаленного режима укажите:
- `transport/type`: `remote`/`http`
- `url`: `http://127.0.0.1:8080/mcp`

Проверка:

```bash
opencode mcp list
```

### Другие MCP клиенты

Если клиент поддерживает stdio-конфиг в формате `mcpServers`, используйте шаблон:

```json
{
  "mcpServers": {
    "aurora-grimoire": {
      "command": "$AURORA_GRIMOIRE_BIN",
      "args": ["mcp", "start", "--stdio"]
    }
  }
}
```

Если клиент поддерживает streamable HTTP, поднимите сервер:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
```

и подключите URL `http://127.0.0.1:8080/mcp`.

## Установка skill/command шаблонов для агентов

В CLI добавлена команда-установщик:

```bash
aurora-grimoire agents install --runtime all --scope global --verbose
```

Параметры:
- `--runtime <claude|opencode|codex|all>` (по умолчанию `all`)
- `--scope <global|local>` (по умолчанию `local`)
- `--config-dir <PATH>` (только для одного runtime)
- `--force` (перезаписать существующие файлы)

Что устанавливается:
- Claude Code: `commands/aurora/search-docs.md` (вызов: `/aurora:search-docs ...`)
- OpenCode: `command/aurora-search-docs.md` (вызов: `/aurora-search-docs ...`)
- Codex: `skills/aurora-search-docs/SKILL.md` (вызов: `$aurora-search-docs ...`)

## Основные команды

```text
aurora-grimoire
├─ agents
│  └─ install
├─ rag
│  ├─ config init
│  ├─ fetch-web
│  ├─ struct
│  ├─ chunk
│  ├─ embed
│  ├─ deploy
│  ├─ bundle (create/inspect/extract)
│  ├─ dev (up/down/status/logs)
│  ├─ test-e2e
│  └─ clear
├─ cli
│  └─ search_docs
└─ mcp
   ├─ start
   └─ smoke
```

Полная и актуальная структура с флагами: [`docs/architecture.md`](docs/architecture.md)

## Полезные команды

Проверка статуса dev stack:

```bash
aurora-grimoire rag dev status
```

Логи:

```bash
aurora-grimoire rag dev logs -f
```

Очистка артефактов:

```bash
aurora-grimoire rag clear --all
```
