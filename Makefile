.PHONY: install lint fmt type test check db seed eval beir financebench run
install: ; uv sync --all-extras
lint:    ; uv run ruff check . && uv run ruff format --check .
fmt:     ; uv run ruff format . && uv run ruff check --fix .
type:    ; uv run mypy
test:    ; uv run pytest -q
check:   lint type test
db:      ; docker compose -f docker/docker-compose.yml up -d postgres
seed:    ; bash evals/seed.sh
eval:    ; uv run ragstack eval evals/golden/handbook.jsonl $(foreach c,$(wildcard evals/configs/*.yaml),--config $(c))
# BEIR benchmark (needs real embeddings: RAGSTACK_EMBEDDER=fastembed RAGSTACK_RERANKER=fastembed)
BEIR ?= scifact
beir:
	python3 evals/beir/prepare.py $(BEIR)
	uv run ragstack user reader --tenant beir
	uv run ragstack ingest .data/beir/$(BEIR)/corpus/*.md --tenant beir --everyone > /dev/null
	uv run ragstack eval .data/beir/$(BEIR)/golden.jsonl --out evals/runs-$(BEIR) $(foreach c,$(wildcard evals/configs/*.yaml),--config $(c))
# Company 10-K PDFs through docparse (FinanceBench subset, page-level evidence + ACL cases)
financebench:
	python3 evals/financebench/prepare.py
	bash .data/financebench/seed.sh
	uv run ragstack eval .data/financebench/golden.jsonl --out evals/runs-financebench $(foreach c,$(wildcard evals/configs/*.yaml),--config $(c))
run:     ; uv run ragstack serve --reload
