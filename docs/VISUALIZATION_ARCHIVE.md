# Historical visualization test: explicitly archived

The original EGO-Det V2 repository contained
`ego_det_v2/tests/test_paper_visualization.py`, which directly imports a
`ego_det_v2.paper_visualization` package that was not shipped in the original
source archive. It was not a passing test in this repository.

During code cleanup we moved that file byte-for-byte to
`archive/legacy_tests/paper_visualization_contracts.py`, with its SHA256 recorded
in `archive/legacy_tests/README.md`. The original test contents remain available
for recovering the plotting/analysis pipeline in future.

The active source audit now treats **all** unresolved imports inside active
source roots as errors; it no longer exempts the missing visualization test.
This is a repository-scope correction, not a change to EGO-Det detection math.

The archive is not imported during normal training, evaluation, or tests.
Existing paper visualization figures should not be claimed reproducible until
those modules and their data inputs have been restored and validated.
