# Archived visualization contracts (not part of active tests)

`paper_visualization_contracts.py` preserves the exact historical contents of
`ego_det_v2/tests/test_paper_visualization.py` from the first GitHub baseline.

SHA256 of the preserved file: `543bdaf151f97123d46b2f92b020fb6dcf082c14fb3493c41f1f19e0ef0a140b`

The test imports `ego_det_v2.paper_visualization.*`, but that package is
**absent from the original source ZIP and the repository**. Running the file
would fail during test collection; its exclusion is explicit and documented.
The file was moved intact, not rewritten or deleted. This is not proof that the
visualization features work. To restore it, recover the original implementation
files, verify the corresponding test contract, then move the test back to the
active test tree.

No model code, evaluation code, or experimental result is affected.
