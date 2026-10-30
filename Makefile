.PHONY: install lint fmt type test check db seed eval run
install: ; uv sync --all-extras
lint:    ; uv run ruff check . && uv run ruff format --check .
fmt:     ; uv run ruff format . && uv run ruff check --fix .
type:    ; uv run mypy
test:    ; uv run pytest -q
check:   lint type test
db:      ; docker compose -f docker/docker-compose.yml up -d postgres
seed:    ; bash evals/seed.sh
eval:    ; uv run ragstack eval evals/golden/handbook.jsonl $(foreach c,$(wildcard evals/configs/*.yaml),--config $(c))
run:     ; uv run ragstack serve --reload
