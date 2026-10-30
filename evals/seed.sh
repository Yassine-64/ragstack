#!/usr/bin/env bash
# Seeds the demo tenant: users + groups, then the corpus with its ACLs. Idempotent enough for demos.
set -euo pipefail
cd "$(dirname "$0")/.."
R="uv run ragstack"
$R user alice --group engineering
$R user bob   --group finance
$R user carol --group hr
$R user dave
$R ingest evals/corpus/public/*.md --everyone "$@"
$R ingest evals/corpus/engineering/*.md --group engineering "$@"
$R ingest evals/corpus/finance/*.md --group finance "$@"
$R ingest evals/corpus/hr/*.md --group hr "$@"
