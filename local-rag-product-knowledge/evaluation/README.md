# Evaluation

## Questions (`questions/`)

One CSV per manual with the columns `id, category, question, expected_answer, expected_file, expected_page`. The expected answers include scoring notes, for example which wrong value to watch for in a variant trap.

## Results (`results/`)

Each test run produces two files, written automatically by `rag.py test`:

- `results_<mode>_<tag>_<timestamp>.csv`: one row per question with the answer, retrieved sources, whether the expected page was retrieved, timings and token counts. The last four columns (`verdict`, `source_correct`, `failure_mode`, `notes`) are for manual scoring.
- `run_info_<mode>_<tag>_<timestamp>.json`: all settings, the system prompt, model versions and quantization, memory use, mean and max response time.

Each manual folder also has an `ingest_*.json` with page and chunk counts and pages without extractable text.

Included per manual: one RAG run at top-k 5, one at top-k 10 and one baseline without retrieval. For baseline runs the `top_k` value in `run_info` is not used. Runs from setup and debugging (including one with top-k accidentally set to 19) are left out.

To rebuild the summary table in the main README:

```bash
python evaluation/summarize.py
```
