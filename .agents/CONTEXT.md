# Слой агентов: Balansis

<!-- СГЕНЕРИРОВАНО xt-agents sync. Правки вносятся в
     /root/.agents/registry/*.yaml, а не в этот файл. -->

Роль проекта: Математическое ядро ACT, формальные доказательства Lean4
Путь: `/root/StudyNinja-Eco/projects/Balansis` · Репозиторий: `StudyLabPro/Balansis`

## Кто здесь работает

| Агент | Вендор | Полномочия |
|---|---|---|
| `balansis@claude-code` | Claude Code | build |
| `balansis@codex` | OpenAI Codex CLI | build |

Все они читают один и тот же набор скиллов и один и тот же контракт.
Разные вендоры — это разные исполнители одной роли, а не разные роли.

## Скиллы этого агента

- **core** — `agent-layer`, `agent-native-comms`, `codebase-memory`, `hot-development-lane`, `humanize-writing`, `no-archived-repositories`, `no-coauthor`, `no-human-time-estimates`, `no-production-data-publication`, `public-repository-policy`
- **infra** — `environment-drift`, `infra-inventory`, `lab-to-pr`, `observability-ops`, `production-access`, `xteam-control-center`, `xteam-infra`
- **method** — `agi-agent`, `full-output-enforcement`, `skill-creator`
- **domain** — `ip-publication-gate`, `magic-architecture`

Уникальны для проекта: `ip-publication-gate`, `magic-architecture`.
Остальное — общий инфраструктурный знаменатель `/root/.agents/skills`;
своей копии проект не заводит.

## Как связаться с другими агентами

```bash
xt-agents roster                       # кто вообще есть
xt-agents whoami                       # кто я в этом каталоге
xt-agents inbox                        # входящие от других агентов
xt-agents send <agent-id> 'текст'      # написать другому
xt-agents handoff --to <agent-id> ...  # передать работу по схеме
```

Канал выбирается по сроку жизни решения:

- **шина** — живой обмен между сессиями на этом хосте; не источник правды;
- **граф** (`codebase-memory-mcp`) — структурное знание о коде и ADR-решения;
- **GitHub** — обязательства и приёмка; доска https://github.com/orgs/StudyLabPro/projects/1.

## Границы

- Прямой push в защищённые ветки запрещён; работа идёт через PR.
- Необратимое действие требует подтверждения владельца.
- Дефект живой среды сначала получает запись, потом правку.
- Секреты не попадают ни в шину, ни в граф, ни в issue.
