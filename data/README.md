# Bundled ACE test data

These files were copied from the local `feasibility/data/` directory without
modifying the released examples:

- `ace_test.json`: 1,100 released test scenarios, under the top-level `scenarios` key.
- `ace_test_source.json`: the identical original download, retained for inspection.
- `provenance.json`: source URL, download timestamp, SHA-256 checksum and license note.

Each scenario includes `clauses` (evidence IDs mapped to text), `scenario_text`
and `gd_tr` (reference classification). Model inputs exclude `gd_tr`.

Attribution: Ayush Singh, Dishank Aggarwal, Pranav Bhagat, Ainulla Khan,
Sameer Malik and Amar Prakash Azad, *COMPACT: Building Compliance Paralegals
via Clause Graph Reasoning over Contracts*, EACL 2026, pp. 8081–8112.

- Paper: https://aclanthology.org/2026.eacl-long.377/
- Dataset publisher: Fujitsu Research
- Source repository: https://github.com/FujitsuResearch/Fujitsu-Assessing-Compliance-in-Enterprise-Dataset
- License: CC BY 4.0, as stated in the source repository README.
- License terms: https://creativecommons.org/licenses/by/4.0/

Source corpora include CUAD and ContractNLI. Preserve source attribution.
This snapshot was downloaded from `main`; its checksum identifies the exact
bytes retained here. It does not include a recorded source commit SHA.

The data evaluates scenario compliance with supplied relevant clauses. It does
not provide before/after revision annotations or full-agreement retrieval labels.
