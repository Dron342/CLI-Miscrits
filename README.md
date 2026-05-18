# CLI Miscrits

Standalone Python client scaffold for operating a Miscrits account without any game client files.

The client talks directly to the Miscrits Nakama HTTP API:

- API host: `https://worldofmiscrits.com`
- CDN host: `https://cdn.worldofmiscrits.com`
- server key: `a1c737cc188f54ab3658ba5da0e12ee5`
- common RPC methods: `get_player`, `heal_team`, `wish_sk`, `wish_vi`, `wish_xmas`, `get_collector_quest`, `create_battle`

`CLI_Miscrits` is intentionally self-contained. You can copy this folder to another machine or an empty directory and run it with Python 3.10+; it does not import or read files from the original game project.

It starts with account/session/RPC tooling and a local web panel. Battle automation can be added on top after the battle socket protocol and payloads are implemented in Python.

## Run

Requires Python 3.10+.

```powershell
cd CLI_Miscrits
python -m miscrits_cli doctor
python -m miscrits_cli version
python -m miscrits_cli check-update
python -m miscrits_cli request-info
python -m miscrits_cli cache-sync
python -m miscrits_cli cache-list
python -m miscrits_cli avatar-sync
python -m miscrits_cli login <username-or-email> <password>
python -m miscrits_cli player
python -m miscrits_cli heal
python -m miscrits_cli wish sk
python -m miscrits_cli breed-plan
python -m miscrits_cli breed-once --dry-run
python -m miscrits_cli auto-breed --max-breeds 10 --dry-run
python -m miscrits_cli serve --host 127.0.0.1 --port 8765
```

The client stores session data in `data/session.json` inside this folder. Do not commit that file if this folder is later added to version control.

Reference data is cached under `data/cache/`. `cache-sync` downloads `cache.json` from the CDN, compares local versions, and refreshes required JSON files such as `miscrits.json`. Some entries are not exposed as direct CDN files; for those, CLI falls back to the same `get_<name>` RPC pattern as the game client, so `cache-sync --all` should be run after login.

Miscrit avatars are cached under `data/cache/assets/avatars/`. The web UI downloads missing avatars lazily through the CLI server, and `avatar-sync` can prefetch saved-account avatars or `avatar-sync --all` can prefetch the full local `miscrits.json` list.

## Notes

- Application versions follow `MAJOR.MINOR.PATCH`. Publish a Git tag such as `v0.2.0` for every released version so `check-update` can compare the local build with the latest published tag.
- `rpc` accepts any known server RPC method:

```powershell
python -m miscrits_cli rpc get_player
python -m miscrits_cli rpc update_location --payload "{\"locationId\":1,\"areaId\":1}"
```

- The web interface exposes the same core actions on `http://127.0.0.1:8765`.
- The implementation uses Python standard library networking so the scaffold runs without installing dependencies.
- HTTP requests use a Godot-style client profile: explicit API port `443`, `GodotEngine/...` user agent, and version headers.

## Изменения в 0.2.1

- Добавлены безопасные повторы для временных сетевых сбоев при `GET`-запросах, включая TLS-обрывы `UNEXPECTED_EOF_WHILE_READING`.
- Ошибка арены `Match not found` теперь считается восстанавливаемой, а не фатальной.
- План больше не завершается целиком из-за временной сетевой ошибки: он переходит в восстановление и продолжает следующий цикл.
- Версия приложения теперь видна в web-интерфейсе до входа и после входа.

## Изменения в 0.3.0

- Добавлено автоматическое обновление CLI до последнего опубликованного Git-тега.
- Перед обновлением web-сервер и прямой запуск `arena-run` мягко останавливают план и циклы арены: новые бои больше не ищутся, а текущий бой дожидается естественного завершения.
- После установки новой версии процесс сам перезапускается и восстанавливает остановленный план либо оставшееся число боёв арены.
- Автообновление выполняется только на чистом Git-checkout'е и только через fast-forward, чтобы не затирать локальные изменения.

## Breeding

The S+ breeding planner uses only level 1 miscrits, skips team members and favorites, and avoids spending unique S+ copies unless explicitly allowed.

```powershell
python -m miscrits_cli breed-plan
python -m miscrits_cli breed-once --dry-run
python -m miscrits_cli breed-once
python -m miscrits_cli auto-breed --max-breeds 10
python -m miscrits_cli breed-plan --target-mid 123 --min-max-sum 18
python -m miscrits_cli auto-breed --target-element Fire --max-breeds 10
```

Useful options:

- `--min-max-sum 18` only accepts triples that can theoretically produce S+.
- `--allow-splus-parents 1` allows spending duplicate S+ parents when needed.
- `--target-mid ID` prefers and requires a breeding plan that can produce that species.
- `--target-element NAME` prefers and requires a breeding plan that can produce that element.
- `--dry-run` prints the selected parents without calling the `breed` RPC.
