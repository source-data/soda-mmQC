---

## title: doc-checklist roadmap — sorting and gap analysis
status: draft
date: 2026-08-21
tags: [doc-checklist, mmqc, benchmarking, aip-guideline]

# doc-checklist roadmap

Working document for sorting existing/proposed doc-checklist checks by
**decision type** (mechanical vs. judgment) and **benchmarkability**, and for
tracking overlap with the AIP QC guideline (shAIdowed).

## How to read this

- **Mechanical**: exact/rule-based, clean gold answer, fits today's flat-leaf
schema + eval-manifest scoring with no new design work.
- **Judgment**: requires model reasoning about quality/appropriateness;
may still be benchmarkable with graded/semantic scoring, but needs real
curation effort and possibly tool use (web/lookup).
- **Report-only**: doesn't force well into a scored leaf; better served as a
free-text section an editor reads, not a pass/fail.

Flags used below:

- ⚠️ **contradiction** — existing check's pass condition may be backwards
relative to the AIP guideline; resolve before benchmarking further.
- 🔀 **merge candidate** — overlaps with another row; should become one
check with multiple leaf fields, not several checks.
- 🚫 **out of scope** — lives in the submission platform / eJP, not in
mmQC's text+figure-image surface.

---



## Document-level checks


| Check                                  | Description                                                | Type                                                | Benchmarkable?  | Notes                                                                                                                                                                                                            |
| -------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `author-contribution-in-ms`            | Manuscript includes an author contribution statement       | mechanical                                          | yes             | ⚠️ AIP guideline (`AC/CRediT`) wants this section **removed**, not present — pass condition may be inverted. Resolve before building further.                                                                    |
| `biorender-protocol.io-mentions`       | BioRender/protocols.io mentions in correct chapter         | mechanical                                          | yes             | ⚠️ AIP guideline wants disclaimers **removed** from legends and consolidated into one Methods line — check the direction of the pass condition here too.                                                         |
| `DAS-present-and-correct`              | DAS present and correctly formatted                        | mechanical + judgment                               | yes (partially) | Really 2–3 checks glued together: presence (mechanical), section placement (overlaps `section_order`), and per-accession URL + reviewer access code (AIP's #1 most common fail case — worth its own leaf field). |
| `data-not-shown`                       | Mentions of "data not shown"                               | mechanical                                          | yes             | Matches AIP `DATA NOT SHOWN` key.                                                                                                                                                                                |
| `external-data-url-validation-agentic` | External data links resolve to the specific entry          | mechanical (presence) + tool-dependent (resolution) | probably        | Presence check is mechanical; actual URL resolution needs a fetch/tool step — not a pure flat-leaf check.                                                                                                        |
| `no-overclaim-in-abstract`             | Abstract contains generalization/overclaim                 | judgment                                            | difficult       | Deep thinking — no external doc reference needed, but "overclaim" is inherently fuzzy/subjective.                                                                                                                |
| `section_order`                        | Manuscript sections in correct order                       | mechanical (nominally)                              | probably        | Own note: easier to describe errors in prose than force a strict schema. Candidate for **report-only**, or benchmarkable only if reduced to a single pass/fail.                                                  |
| `AB-target-reagent-consistency`        | Antibody in Methods actually targets the protein described | judgment                                            | small set maybe | Own note says "ideally with deep thinking and web search" — relabel from mechanical to judgment - not sure yet.                                                                                                  |
| Figure callouts                        | —                                                          | mechanical                                          | yes             | Existing Python tool to give to agent; matches AIP `FIGURE CALLOUTS` key directly.                                                                                                                               |




### From LSA reporting requirements


| Check                         | Description                                               | Type                  | Benchmarkable? | Notes                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------- | --------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DNA Sequences                 | Primer/oligo/plasmid/RNAi/CRISPR sequences provided       | mechanical (presence) | yes            | Presence check mechanical; correctness/deeper verification would need web search + judgment — split into two checks if pursued.                                                             |
| Antibodies                    | Catalog numbers + concentration provided                  | mechanical (presence) | yes            | 🔀 merge candidate — presence half overlaps `AB-target-reagent-consistency`. Keep presence as its own mechanical `reagents-reported` check; let target-consistency stand alone as judgment. |
| Western blotting              | Acquisition/quantification method stated                  | mechanical            | yes            | Simple in-document check.                                                                                                                                                                   |
| Microscopy                    | Microscope/objective/camera/software details provided     | mechanical            | yes            | Simple in-document check.                                                                                                                                                                   |
| Methods description           | Methods comprehensive, not just referencing a prior paper | judgment              | no             | Best **report-only** candidate — genuinely resists pass/fail; write as prose flag instead.                                                                                                  |
| Molecular weight markers      | ??                                                        | —                     | —              | Undefined — needs scoping.                                                                                                                                                                  |
| Proteomics / structural omics | ??                                                        | —                     | —              | Undefined — needs scoping.                                                                                                                                                                  |


---



## Reference-list checks


| Check                           | Description                                                         | Type                            | Benchmarkable?             | Notes                                                                                                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------- | ------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reference-no-more-than-10`     | ≤10 authors before "et al."                                         | mechanical                      | yes                        | 🔀 merge candidate — fold into one `reference-format` check as a leaf field, not a standalone check.                                                                                          |
| `reference-no-doi`              | No DOIs in reference list                                           | mechanical                      | yes                        | 🔀 same merge.                                                                                                                                                                                |
| `reference-format-alphabetical` | Alphabetical order                                                  | mechanical                      | yes                        | 🔀 same merge — these three become one check (`reference-format`) with three leaf fields: `author_count_ok`, `doi_absent`, `alphabetical_order_ok`.                                           |
| `citation-supports-claim`       | Reference backs the claim in its sentence/paragraph                 | judgment                        | hard (currently a "dream") | Needs a fetch step (retrieve cited paper's abstract/full text) before real benchmarking is possible. See separate design note.                                                                |
| `self-citation`                 | Unnecessary self-citation in reference list                         | judgment (partially mechanical) | maybe                      | The *ratio* of self-citations is a countable, mechanical pre-step; only "is it *unnecessary*" is real judgment. Consider splitting into `self-citation-ratio` (mechanical) + a judgment flag. |
| `in-text-citation-matches-list` | Every in-text citation exists in the reference list, and vice versa | mechanical                      | yes                        | Simple in-document check.                                                                                                                                                                     |
| `reference-style-consistency`   | Same citation style throughout (not mixing numbered/author-year)    | mechanical                      | yes                        | Could replace/absorb `reference-format-alphabetical`.                                                                                                                                         |
| `retracted-reference`           | Reference points to a retracted paper                               | judgment (easy)                 | yes, with a lookup tool    | Relabel from mechanical — needs a retraction-database lookup, but the judgment itself is a simple binary fact-check, not fuzzy interpretation.                                                |


---



## Gap analysis vs. the AIP (accepted-in-principle) QC guideline

Checked against the shAIdowed AIP guideline. Not everything is covered by
fig-checklist or doc-checklist as assumed — see breakdown below.

### ⚠️ Contradictions to resolve first

1. `author-contribution-in-ms` vs. AIP's `AC/CRediT` — existing check
  scores presence as pass; guideline wants the free-text section **removed**
   in favor of structured CRediT entries in the submission system.
2. `biorender-protocol.io-mentions` vs. AIP's BioRender handling —
  guideline wants disclaimers removed from individual figure legends and
   consolidated into one Methods line, not just "mentioned in the right
   chapter."



### Genuinely covered (real overlap, safe to treat as duplicate)

- Reference list checks (alphabetical, ≤10 authors, DOI removal) → AIP `REFERENCES`
- `data-not-shown` → AIP `DATA NOT SHOWN`
- Figure callouts → AIP `FIGURE CALLOUTS`
- Figure-legend statistics (p-values, stat test, n, error bars, scale bars) → fig-checklist territory, matches AIP's `*Figure Legends - Comments*` section



### Missing — new mechanical, text-based checks to add

- `COI/DCIS` — section named "Disclosure and Competing Interests Statement" (not "Conflict of Interest"/"Ethics declarations"), placed after Acknowledgments.
- `ORCID ID` — present for every corresponding author.
- `AFFILIATIONS` — listed on title page, not in footnotes.
- Section renaming conventions — "Materials and Methods" → "Methods", "Summary" → "Abstract", "Abbreviations" section removed (defined on first use instead).



### 🚫 Out of scope for mmQC (submission platform / package-level, not text or figure-image)

These need file-manifest inspection or image metadata mmQC's text+figure-image
architecture doesn't cover — explicitly scoped out, not silently dropped:

- `FIGURES IN SEPARATE FILES`
- `MOVIES` (individually zipped, per-movie legend file)
- `SYNOPSIS IMAGE` (file format, pixel dimensions)
- `APPENDIX FILE WITH ToC`
- `SOURCE DATA` (folder structure)
- `R&T TABLE` (separate file or not)



### 🚫 Out of scope for mmQC (needs external system data — eJP)

Requires comparing the manuscript against data that lives in the submission
system, not just reading the manuscript alone — open question whether this
is reachable at all without a connector:

- `AUTHORS` (names match eJP exactly)
- `FUNDING INFO` (eJP entries match manuscript-acknowledged funders)

---



## Suggested merges (reduce check count before building)

1. `reference-no-more-than-10` **+** `reference-no-doi` **+** `reference-format-alphabetical` → one `reference-format` check, three leaf fields, one gold file instead of three.
2. **Antibodies (LSA, presence)** + `AB-target-reagent-consistency` → split into `reagents-reported` (mechanical presence) and keep `AB-target-reagent-consistency` standalone (judgment + web lookup).
3. `DAS-present-and-correct` → split into presence/format (mechanical) and section-placement (folds into `section_order`).



## Open questions to raise

- Is an eJP connector realistically on the roadmap, or are `AUTHORS`/`FUNDING INFO` permanently out of scope?
- Does `section_order` become a report-only check, or can it be reduced to a single benchmarkable pass/fail?
- Resolve the two ⚠️ contradictions above before curating gold data for those checks.

