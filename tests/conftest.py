"""
Pytest-wide setup that must run before any test module imports `src`.

WHY THIS FILE EXISTS: src/ is instrumented with Opik (@opik.track on
generate_answer, query_collection, rerank, plan_query). The test suite drives
that pipeline hundreds of times against mocked LLMs and a throwaway index —
none of which are real answers worth keeping, and all of which would land in
the same Tech-Doc-Assistant project you actually read traces from, burying the
real ones. OPIK_TRACK_DISABLE turns every decorator into a pass-through, so
tests also never need an Opik API key or network access.

conftest.py is imported by pytest before it collects test modules, which is
what makes setting the variable here (rather than in a fixture) effective —
by fixture time the `src` modules have already been imported.

.github/workflows/ci.yml sets the same variable at the workflow level.
"""

import os

os.environ.setdefault("OPIK_TRACK_DISABLE", "true")
