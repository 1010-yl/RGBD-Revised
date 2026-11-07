# Response to Reviewers — working draft (NOT submission-ready)

## Comment 1: Relationship to UMFNet, missing citation, closely corresponding framework and equations

**Response draft (replace bracketed statements only after implementation and validation):**

We thank the reviewer for identifying the insufficient attribution. We agree that the original manuscript did not clearly distinguish its technical contribution from Wang et al., *Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection* (CVPR 2026). We have [to be completed: explicitly cited and discussed the work in the Introduction and Related Work, distinguished what was reused, and revised all overlapping contribution claims].

Our revised research direction targets the reliability of registered RGB-D measurements rather than implicit alignment of unregistered RGB-T features. We [to be verified after final implementation: replaced the previous alignment-based module with RGB-structural-prior-conditioned depth uncertainty modeling; redesigned the fixed channel/spatial modulation into a difficulty-aware progressive fusion mechanism; and changed the information flow of the encoder–decoder]. The original UMFNet's methods, formulas, and figures are acknowledged as prior work, and we will not claim those elements as original contributions.

**Evidence to attach before submission:** new source/code diff, new architecture figure, equation-to-implementation mapping, ablations, robustness plots, direct citation.

## Comment 2: Meaning of the ± notation in Table 1

We agree that the statistical reporting was insufficient. We will [identify the actual origin of each ± value and specify whether it is the standard deviation, standard error, or confidence interval]. If these values cannot be traced to reproducible measurements, we will replace them with verified point estimates and avoid unsupported uncertainty claims. Where repeat training is available, we will report the number of independent seeds and the relevant formula.

**Do not invent runs or describe ± as std without evidence.**

## Comment 3: Qualitative figure does not match cited scenes

We acknowledge the mismatch between the text and the qualitative examples. We will reconcile every named scene with the actual RGB/Depth/GT/prediction file stems, replace inaccurate descriptions, and ensure that the figure displays real prediction outputs obtained using the reported checkpoints.

## Comment 4: Wrong attribution for weighted F-measure and E-measure

We will verify the original references and correct the in-text citations and bibliography, ensuring that the citation for weighted F-measure points to its source and the citation for E-measure points to the enhanced-alignment measure work.

## Comment 5: Repeated [0,1]

We will remove the duplicated instances and retain one mathematically correct range.

## Comment 6: Fig. 2 caption repeated title

We will remove the redundant "Fig. 2. Figure 2." prefix and review all figure caption numbering.

## Comment 7: Placeholder running head

We will replace every occurrence of "Short Title for Running Head" with the correct short title before final typesetting.

## Submission checklist

- [ ] Rebuttal letter contains only completed actions; remove all planning language and brackets.
- [ ] Tracked-changes version matches clean version.
- [ ] Figure numbering and cross-references are consistent across both manuscript versions.
- [ ] Experimental statistics are traceable to logs/checkpoints.
- [ ] Complete citations and code provenance are disclosed.
- [ ] Final experiment figures show actual samples, not conceptual illustration.
