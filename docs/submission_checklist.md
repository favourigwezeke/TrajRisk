# TrajRisk — Final manuscript and submission checklist

Updated 20 October 2026 after Repository Check 2 and frozen-model replay, using the active canonical LaTeX and the final review completed 5 October. The master remains **TrajRisk_Combined_Review.pdf**, compiled from **review_all.tex**. The approved manuscript has **64 pages, a 178-word abstract, 126 bibliography entries, 26 numbered equations, seven main figures, five supplementary figures, six supplementary tables, and five supplementary data files**.

Checked items describe the completed canonical manuscript work. They do not certify a new independent rerun, author approval, public repository release, or submission. `N/A` items are omitted from the current LaTeX workflow. Two verification tasks were explicitly waived by the user: recovery of the original DE/pathway-settings generator and exact reproduction of six precision-sensitive historical Matchett statistics. Their retained results are not called independently reproduced. R 4.5.1 / Monocle 3 1.4.27 was confirmed by the user.

| Section | Current status |
| --- | --- |
| 1. References | Canonical bibliography locked; current preprint refresh and exact PLOS reference formatting remain submission tasks |
| 2. Supplementary text | Complete |
| 3. Numerical audit | Complete within the two recorded verification exceptions |
| 4. Methods | Complete within the recorded scope |
| 5. Cross-references | Complete |
| 6. Terminology | Complete |
| 7. Equations | Complete |
| 8. Legends and captions | Complete |
| 9. Discussion | Complete |
| 10. Abstract | Complete, 178 words |
| 11. Front matter and declarations | Title present; author details and declarations pending |
| 12. Repository | In progress: frozen replay passes 79 checks in an isolated environment; S1–S5 regenerate; final main-figure alignment, remaining tables and public release pending |
| 13. Data package | Data S1–S5 checked and packaged; final deposition link pending |
| 14. File QA | Canonical PDF/source checks complete; submission-file checks remain |
| 15. Scientific consistency | Substantive interpretation checks complete; final read of author-completed submission pending |
| 16. Submission instructions | Current PLOS instructions reviewed 20 October; application to the final submission still pending |
| 17. Submission materials | Canonical body/assets ready; author summary, cover letter, declarations and links pending |
| 18. Release | Pending |

## Next working sequence

1. Source/environment and frozen-input collection are complete. The saved-model replay passes 79 checks; all five supplementary generators run.
2. Complete final main-figure generator alignment and remaining table reproduction, then prepare the versioned data/model archive.
3. Fill author-approved front matter and declarations; draft the author summary and cover letter.
4. Complete submission formatting, final read, exact version freeze and upload preview.

The verified code is maintained in the private TrajRisk GitHub repository; the submitted-version tag and archival release remain pending. PyPI packaging is optional and can follow the preprint; it does not block the current repository work. Checks 1 and 2 are complete. The source laptop’s Data/data alias is verified. Frozen inference and supplementary plotting have been run; raw-data retraining and R reruns are separate.

## 1. REFERENCE AUDIT

- [x] Extract every in-text citation from the manuscript.

- [x] Extract the complete bibliography/reference list.

- [x] Check that every in-text citation has exactly one matching bibliography entry.

- [x] Check that every bibliography entry is actually cited in the manuscript.

- [x] Remove duplicate references.

- [x] Detect references that are the same paper but entered differently.

- [x] Check author names, title, journal/conference, year, volume, issue, pages/article number.

- [x] Verify DOI where one exists.

- N/A — Full PMID/PMCID verification was explicitly not requested.

- [ ] Finish a current published-version sweep for the seven remaining standalone preprint entries. Earlier selected publication updates are complete; the 20 October search did not establish a replacement for every record. Keep the canonical entries until a replacement is verified.

- [x] Check whether conference/workshop papers should be cited as proceedings papers rather than URLs/preprints.

- [x] Check software/method references: scVI, Scanpy, Monocle 3, PyTorch Geometric, XGBoost, etc.

- [x] Check dataset/source-paper references for Walesky, Umbaugh, Ben-Moshe, Matchett, and all other external datasets.

- [x] Check that biological claims are supported by the cited papers rather than merely adjacent literature.

- [x] Check citation placement sentence by sentence: citation should support the claim immediately preceding it.

- [x] Remove citations that do not actually support the sentence.

- [x] Add genuinely missing citations only where necessary.

- [ ] Apply/confirm exact PLOS Vancouver reference formatting in the submission version. Canonical numbering and bibliography matching are complete.

- [x] Final in-text citation ↔ bibliography 1:1 audit.

- [x] Lock the 126-entry canonical bibliography, preserving the pending submission refresh/style tasks above.

## 2. FINALIZE SUPPLEMENTARY INFORMATION TEXT

- [x] Final Figure S1 legend.

- [x] Final Figure S2 legend.

- [x] Final Figure S3 legend.

- [x] Final Figure S4 legend.

- [x] Final Figure S5 legend.

- [x] Final Table S1 title/caption/notes.

- [x] Final Table S2 title/caption/notes.

- [x] Final Table S3 title/caption/notes.

- [x] Final Table S4 title/caption/notes.

- [x] Final Table S5 title/caption/notes.

- [x] Final Table S6 title/caption/notes.

- [x] Final Data S1 description.

- [x] Final Data S2 description.

- [x] Final Data S3 description.

- [x] Final Data S4 description.

- [x] Final Data S5 description.

- [x] Make sure every abbreviation appearing only in the supplement is defined.

- [x] Make sure statistical tests, directions, thresholds, denominators, and units are explicit.

- [x] Lock Supplementary Information text.

## 3. NUMERICAL CLAIM AUDIT — MANUSCRIPT AGAINST LOCKED SOURCES

- [x] Audit every `n`.

- [x] Audit every AUROC.

- [x] Audit every AUPRC.

- [x] Audit every confidence interval.

- [x] Audit p-values within the agreed scope; six exact precision-sensitive historical Matchett fields remain waived.

- [x] Audit Spearman/Pearson coefficients within the same recorded scope.

- [x] Audit every odds ratio.

- [x] Audit every percentage.

- [x] Audit every mean/median.

- [x] Audit every gene count.

- [x] Audit every cell/nucleus count.

- [x] Audit every edge count.

- [x] Audit every donor count.

- [x] Audit every dimensionality/latent-space number.

- [x] Audit every model parameter/hyperparameter stated in Results or Methods.

- [x] Audit every value against the locked table/data/file, not memory.

- [x] Verify rounding is consistent between text, figures, and tables.

- [x] Lock the canonical Results numbers within the documented verification scope.

## 4. FINAL METHODS CORRECTIONS

- [x] Replace mouse external-transfer Methods with the final audited Reference-KNN/W_self-only wording.

- [x] Replace human transfer Methods with the final audited Reference-KNN/W_self-only wording.

- [x] Ensure there is no unsupported tie-breaking claim.

- [x] Ensure Reference-KNN is described as reference projection, not human trajectory inference.

- [x] State exact `K = 15`.

- [x] State correct temporal scale mappings.

- [x] State Gaussian similarity weighting correctly.

- [x] State why RF transition weights were not used for external cells.

- [x] Add the Matchett recovery-signature correction: *Cyp2e1/CYP2E1* excluded to avoid circularity with zone definition.

- [x] Verify spatial statistical tests/directions.

- [x] Verify bottleneck threshold definitions.

- [x] Verify quartile perturbation definition.

- [x] Verify cross-species ortholog filtering and 13,122 shared genes.

- [x] Verify software/version statements.

- [x] Verify BioMart wording against the manual export and implemented filters; omit an unverified historical release or assembly.

- [x] Accept the user-confirmed actual RStudio versions: R 4.5.1 and Monocle 3 1.4.27.

- [x] Lock Methods.

## 5. CROSS-REFERENCE AUDIT

- [x] Figures 1–7 all cited correctly.

- [x] Figures S1–S5 all cited correctly.

- [x] Tables S1–S6 all cited correctly.

- [x] Data S1–S5 all cited appropriately.

- [x] No obsolete Figure S7 references.

- [x] No references to retired supplementary plots.

- [x] No panel mismatch, e.g. “Figure S4B” when the result is actually S4C.

- [x] No wrong table subsection, e.g. S4B vs S4C.

- [x] Every figure/table/data citation appears after the claim it supports.

- [x] Do not over-cite Data S files where they add nothing.

- [x] Final figure/table/data citation audit.

## 6. TERMINOLOGY & NOMENCLATURE AUDIT

- [x] `TrajRisk` used consistently.

- [x] `W_self-only` vs `W_self` standardized.

- [x] `Reference-KNN` standardized.

- [x] `APAP-ALF` standardized.

- [x] `Scale 1 — Acute Stress` standardized.

- [x] `Scale 2 — Peak Injury` standardized.

- [x] `Scale 3 — Fate Decision` standardized.

- [x] `Scale 4 — Resolution` standardized.

- [x] “persistent compensation” used consistently.

- [x] “risk signature”, “recovery signature”, and “net signature” never conflated.

- [x] Mouse gene symbols italicized appropriately, e.g. *Cyp2e1*.

- [x] Human gene symbols italicized appropriately, e.g. *CYP2E1*.

- [x] scVI/scRNA-seq/snRNA-seq terminology standardized.

- [x] Cell vs nucleus/nuclei terminology correct by dataset.

- [x] Pseudotime terminology standardized.

- [x] “bottleneck edge”, “bottleneck source cell”, etc. used consistently.

- [x] Hyphens/en dashes/timepoint formatting standardized: `24 h`, `48 h`, `24 h → 48 h`, etc.

## 7. EQUATIONS, SYMBOLS, AND MATHEMATICAL NOTATION — IMPORTANT

- [x] Reconstruct the final equations from the latest code, not old drafts.

- [x] Verify every equation against implementation.

- [x] Verify graph notation.

- [x] Verify node-feature notation.

- [x] Verify directed-edge notation.

- [x] Verify transition-matrix notation.

- [x] Verify RF/Gaussian transition-weight notation.

- [x] Verify message-passing equation.

- [x] Verify scale-specific convolution equation.

- [x] Verify concatenation of four 64-D embeddings → 256-D representation.

- [x] Verify prediction-head notation.

- [x] Verify risk-seed notation.

- [x] Verify damping/propagation equations where relevant.

- [x] Verify flux equation.

- [x] Verify amplification equation.

- [x] Verify bottleneck definition.

- [x] Verify perturbation/ablation notation.

- [x] Define every symbol immediately after first use.

- [x] Ensure the same symbol never means two different things.

- [x] Ensure matrix/vector/scalar typography is internally consistent.

- [x] Ensure indices `i`, `j`, `s`, `t`, etc. are used consistently.

- [x] Remove obsolete equations left from earlier versions.

- [x] Produce publication-quality LaTeX equations; Word export is not part of this workflow.

- [x] Lock mathematical notation.

## 8. MAIN FIGURE LEGENDS / TABLE CAPTIONS / IN-TEXT DISPLAY REFERENCES — FINAL PASS

- [x] ~~Main figure citations substantially completed.~~

- [x] ~~Supplementary figure citations substantially completed.~~

- [x] ~~Supplementary table/data citations substantially completed.~~

- [x] Final legends for Figures 1–7.

- [x] Check every panel A/B/C/etc. against the final rendered figure.

- [x] Check units.

- [x] Check sample sizes.

- [x] Check statistical-test descriptions.

- [x] Check color/group descriptions.

- [x] Ensure legends explain error bars/CI/boxplot components where applicable.

- [x] Ensure no legend claims something not actually plotted.

- [x] Lock display-item captions.

## 9. DISCUSSION / INTERPRETATION PASS

- [x] Separate demonstrated results from hypotheses.

- [x] Keep Scale 3 “resolution gateway” explicitly hypothesis-generating.

- [x] Check claims of mechanism vs association.

- [x] Check claims of conservation across species.

- [x] Check claims of generalization.

- [x] Make limitations explicit where necessary.

- [x] Remove overclaiming.

- [x] Make biological significance clear: what does this enable for liver injury biology/drug safety?

- [x] Final Discussion coherence pass.

## 10. ABSTRACT — WRITE NEAR THE END

- [x] Only after Results/Methods/numbers are locked.

- [x] Background/problem.

- [x] What TrajRisk does.

- [x] Core mouse result.

- [x] External validation.

- [x] Human transfer.

- [x] Spatial validation.

- [x] Bottleneck interpretation.

- [x] Biological/methodological significance.

- [x] Check every numerical value in abstract against locked manuscript.

- [x] Check the abstract against the requested 200-word maximum; the final abstract has 178 words. Recheck journal instructions at submission.

- [x] Lock abstract.

## 11. TITLE / FRONT MATTER

- [x] Final manuscript title present; sentence-case formatting remains part of the PLOS submission pass.

- [ ] Author names.

- [ ] Author order.

- [ ] Affiliations.

- [ ] Corresponding author.

- [ ] Email addresses.

- [ ] ORCIDs.

- [ ] Keywords.

- [ ] Add the PLOS short title (maximum 70 characters).

- [ ] Author contributions.

- [ ] Acknowledgments.

- [ ] Funding.

- [ ] Competing interests/declaration of interests.

- [ ] Ethics statements where applicable.

- [ ] Finalize the data availability statement with complete accession and deposition links; source accessions are already listed.

- [ ] Replace the current code-availability `[X]` with the released repository/archive URL.

## 12. GITHUB / REPRODUCIBILITY REPOSITORY

- [x] Exclude OLD/backup candidates from the repository draft; ten backup-named files are excluded; earlier transfer/full-figure sources are archived. Six audited core scripts, preprocessing sources and five supplementary generators are supplied. Final main-figure alignment is pending.

- [ ] Ensure final scripts reproduce Figures 1–7.

- [x] Ensure final scripts reproduce Figures S1–S5 from saved metadata/coordinates and CSVs; canonical figure PDFs remain unchanged. Exact font rendering depends on the installed fonts.

- [ ] Complete table reproduction. Table S3A endpoints, S4B score endpoints and selected S6/graph/conservation quantities replay successfully; the full table-generating pipeline remains pending.

- [x] Include observed direct-package versions, all 214 Python distributions and 591 Conda entries. Keep the user-confirmed RStudio R 4.5.1/Monocle 3 1.4.27 separate from current Conda R 4.5.3.

- [x] Include `environment.yml`, `requirements.txt`, `pip-observed.txt` and `conda-observed.json` with exact recorded versions, including joblib 1.5.3. The broader training environment is not clean-install tested. A separate frozen-runtime requirements file and tested Linux/Python 3.12 package lock are included.

- [x] Include README with the supported workflow and remaining input/generator requirements; end-to-end execution is still pending.

- [x] Document manuscript source identifiers and retained cohorts; complete raw-file acquisition manifest is pending.

- [x] Collect and document all ten preprocessing/integration/pseudotime notebooks (97 code cells) and the R trajectory source. Clean raw-data execution remains to be tested.

- [x] Document the supplied publication outputs and the remaining reproduction stages.

- [x] Use portable paths in the 37 retained sources. Core and notebook calculation logic is preserved; supplementary readers accept compact saved inputs, S5 normalizes the Q4 display endpoint in memory, and S1 limits scatter tick density for legibility.

- [x] Scan the current draft for common embedded credentials and personal filesystem paths; repeat on newly collected sources before release.

- [x] Exclude stale previews, historical snapshots, audit packets and duplicate figure versions from the draft.

- [ ] Tag/release the exact submitted version.

- [ ] Consider archival DOI/versioned release if appropriate.

- [x] Test the frozen-model workflow in a newly created isolated Linux/Python 3.12 environment; dependency checks pass. Clean raw-expression preprocessing/training remains a separate pending test.
- [x] Receive and review Repository Check 1: 47 exported source hashes verify, six of six audited core sources match, zero collection omissions/errors.
- [x] Add exact observed dependency records: joblib 1.5.3, 214 Python distributions and 591 Conda entries.
- [x] Receive and verify Repository Check 2: all 135 file hashes and six compact objects; no required files missing; Data/data alias confirmed.
- [x] Replay frozen models and numerical results: 79 checks pass, including original mouse bootstrap CIs and human primary/sensitivity donor endpoints.
- [ ] Deposit the final data/model archive with checksums and a public versioned URL.
- [x] Export final frozen Figure 5B mouse scores and corrected distinct-human Figure 6E counts; preserve the canonical PDFs. Full main-figure assembly remains pending.
- [x] Align the Table S6F CSV endpoint with its PDF/caption (1.000); other fields and original uploaded inputs remain unchanged.
- [ ] Choose an author-approved software license and add complete citation metadata.

## 13. FINAL DATA PACKAGE

- [x] Data S1 final.

- [x] Data S2 final.

- [x] Data S3 final.

- [x] Data S4 final.

- [x] Data S5 final.

- [x] Column names understandable.

- [x] Units defined.

- [x] Missing-value conventions defined.

- [x] CSV files open correctly; spreadsheet-only hidden formatting is N/A.

- [x] No accidental index columns.

- [x] No personal/local filesystem information.

- [x] Filenames match manuscript references exactly.
- [ ] Add the final versioned data-archive URL and release identifier.

## 14. FINAL FILE / FORMAT QA

- N/A — DOCX is no longer the canonical format; the LaTeX/PDF workflow is in use.

- [x] Times-compatible typography and heading hierarchy checked in the canonical PDF.

- [x] All 26 equations render correctly in the canonical PDF.

- N/A — Word export is not required.

- [ ] Verify the final journal-upload figure formats and resolution; canonical vector PDFs are present.

- [ ] Verify the final journal-upload supplementary figure formats; canonical vector PDFs are present.

- [x] No clipped legends/panel labels in the completed canonical visual review.

- [x] One active version per figure in the manuscript/repository draft.

- [x] Typeset tables and 20 original CSV components checked.

- [x] All five supplementary CSVs open correctly and match their recorded structures.

- [x] Canonical PDF page breaks reviewed.

- N/A — Word Track Changes is not part of this workflow.

- [ ] Remove private editorial comments from final submission sources where needed.

- [ ] Confirm no private/hidden editorial material in the final submission package.

- [x] Completed 64-page canonical visual review recorded on 5 October; repeat affected-page review after later submission edits.
- [x] Rebuild the bundled preview from the active source, replacing the obsolete 62-page preview.

## 15. FINAL SCIENTIFIC CONSISTENCY PASS

- [ ] Read manuscript from title to final sentence without editing as you go.

- [x] Ask: does the story progress logically? — addressed during the completed interpretation pass.

- [x] Ask: does every major claim have evidence? — addressed during the completed interpretation pass.

- [x] Ask: does every experiment answer a clear question? — addressed during the completed interpretation pass.

- [x] Ask: are negative/null findings represented honestly? — addressed during the completed interpretation pass.

- [x] Ask: are sensitivity analyses clearly separated from primary analyses? — addressed during the completed interpretation pass.

- [ ] Ask: could a reviewer reproduce the analytical choices from Methods?

- [ ] Final consistency pass on the author-completed submission and released repository links.

## 16. JOURNAL-SPECIFIC SUBMISSION CHECK

- [x] Review current PLOS Computational Biology instructions — checked 20 October 2026; re-open immediately before submission.

- [ ] Verify current article structure.

- [ ] Verify current word limits.

- [ ] Verify figure limits.

- [ ] Verify supplementary-file requirements.

- [ ] Verify graphical abstract/highlights requirements if applicable.

- [ ] Verify reporting/checklist requirements if applicable.

- [ ] Verify reference format.

- [ ] Verify data/code deposition requirements.

- [ ] Verify cover-letter requirements.

- [ ] Do not rely on an old cached version of the journal instructions.
- [ ] Prepare a PLOS author summary (150–200 words) distinct from the abstract.
- [ ] Prepare the PLOS submission layout and reference styling while retaining the agreed combined review master.

Current official guidance: https://journals.plos.org/ploscompbiol/s/submission-guidelines . Initial submissions may be a combined PDF and are not required to undergo formatting changes for editorial assessment. The general layout guidance specifies single-column, double-spaced text and continuous line numbers; the abstract maximum is 300 words. The stricter requested 200-word ceiling remains satisfied. Do not treat optional submission packaging as a reason to regenerate the scientific analyses.

## 17. SUBMISSION MATERIALS

- [ ] Final manuscript.

- [x] Main Figures 1–7: canonical active vector assets prepared.

- [x] Supplementary Information content prepared; final upload packaging remains in section 16.

- [x] Tables S1–S6: typeset content and 20 CSV components prepared.

- [x] Data S1–S5: final CSVs and definitions prepared.

- [ ] Cover letter.

- [ ] Highlights/key points if required.

- [ ] Graphical abstract if required.

- [ ] Reporting checklist(s) if required.

- [ ] Suggested reviewers if required.

- [ ] Opposed reviewers if appropriate and permitted.

- [ ] Data/code repository links.

- [ ] Author declarations.

- [ ] Final submission preview inspected before clicking submit.
- [ ] PLOS author summary.

## 18. FINAL RELEASE

- [ ] Freeze manuscript version.

- [ ] Freeze supplement version.

- [ ] Freeze repository commit/tag.

- [ ] Save an exact copy of everything submitted.

- [ ] Record submission date/version.

- [ ] Submit.
