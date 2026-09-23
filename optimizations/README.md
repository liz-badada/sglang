# Optimization ledger

One numbered directory per optimization. An entry is only complete when every
file below exists and COMPARISON.md shows a reproducible end-to-end gain.

    NN-<short-name>/
      BEFORE_RUN_TAG        control run identifier
      AFTER_RUN_TAG         variant run identifier
      before_*.per_kernel.csv   reduction of the control trace
      after_*.per_kernel.csv    reduction of the variant trace
      COMPARISON.md         kernel before/after, end-to-end before/after, accuracy gates
      PR_BODY.md            the PR description for this single change

Bar for an entry to be committed and opened as a PR:
  - a real model-execution kernel change, not a parameter or config change
  - reproducible across paired adjacent arms on one node
  - quantified: us/call for the kernel, us/step end to end, both before and after
  - accuracy gates unchanged, measured as a pair against the control arm

Raw traces are not committed. They are hundreds of megabytes for evidence that
reduces to a few kilobytes; COMPARISON.md records their identifiers so a reader
can regenerate the reduction from the original run.
