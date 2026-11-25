# Documents

The PDF manuals are not included in this repository. They are public and can be downloaded from the publishers:

| Manual | File name used in the tests | Pages | Source |
|---|---|---|---|
| Labkotec idOil Solar user manual (EN) | `DOC1599_idOil-Solar-User-Manual-EN.pdf` | 62 | https://labkotec.fi/wp-content/uploads/DOC1599_idOil-Solar-User-Manual-EN.pdf |
| Nokeval 2021 panel meter manual | `2021-Manual-04072011.pdf` | 20 | https://nokeval.com/app/uploads/2021/01/2021-Manual-04072011.pdf |
| Vaisala K-PATENTS PR-23 Series User Guide, revision H (EN) | `PR-23 Series User Guide IM-EN-PR23-2H.pdf` | 270 | Vaisala PR-23 manuals page: https://vaisala.com/process-refractometer-pr-23-manuals-safety-instructions-and-safety-videos |

Put each manual in its own subfolder (for example `docs/nokeval/`) and build a separate collection for it, so that the test questions for one manual only search that manual:

```bash
python rag.py --collection nokeval ingest --docs docs/nokeval
```

Note: the expected page numbers in `evaluation/questions/` refer to the exact revisions above. Vaisala updates the PR-23 guide regularly, so a newer revision may have different page numbers.
