# Support Ticket Router, Evaluation Report

**Author:** previous engineer on the triage project
**Status:** handed over, awaiting sign off to ship

## Approach

Support messages arrive in a single queue and get routed by hand to one of four
teams: `account-access`, `transaction-dispute`, `fraud-report`, `general`. We
have 400 historical messages labelled with the route they were sent to.

I vectorised the messages with TF-IDF over unigrams and bigrams and trained a
logistic regression on top. I held out 20% of the data as a test set to check
performance.

Run it with:

```bash
python3 baseline/baseline_classifier.py
```

## Result

```
loaded 400 rows
test accuracy: 0.9875
```

**98.75% accuracy on held out data.** Only one message in the test set was
routed incorrectly.

## Class distribution

For reference, the label counts in the training data are:

| route | count |
|---|---|
| general | 160 |
| account-access | 100 |
| transaction-dispute | 90 |
| fraud-report | 50 |

`general` is the most common route and `fraud-report` the least common. The
model reaches 98.75% accuracy regardless, so the distribution does not appear to
be causing a problem in practice.

## Conclusion

The model is accurate, it trains in under a second, and it has no runtime
dependencies beyond scikit-learn. I see no blockers. Recommend we ship this to
production behind the existing queue and revisit only if accuracy degrades.

Possible follow ups, none of which I consider urgent:

- Try a gradient boosting model to see if it beats 98.75%.
- Consider an LLM if we ever add more routes.

## Revised evaluation (added on review)

**Do not sign off on the 98.75% number above.** Two methodology problems:

1. **TF-IDF leakage.** The vectorizer was fit (`fit_transform`) on the full
   corpus *before* the train/test split, so IDF weights were learned partly
   from the held-out rows. The test set wasn't truly unseen.
2. **Single random split, no stratification.** One 80/20 split means the
   15 `fraud-report` examples that happened to land in the test set decide
   the whole story. That's too few to trust, and there's no variance
   estimate.

**Concrete evidence this isn't academic.** Re-running the original code as
written, the single misrouted test message was:

> `fraud-report -> general | "Hi, My account shows activity at 3am that
> wasn't me, $10,000 of USDC is missing."`

That's the single most expensive failure mode this router can make — a live
theft report silently landed in the general queue — and the original report
waved it off as "only one message was routed incorrectly." Real
`fraud-report` recall on that split was 14/15 = 93.3%, not 98.75%.

**Fix applied:** `baseline_classifier.py` now runs a stratified 5-fold CV,
fitting the vectorizer on each fold's train rows only (no leakage), and
reports per-class precision/recall/F1 and a confusion matrix instead of a
single accuracy number.

**Result after the fix:** every class, including `fraud-report`, scores
100% precision/recall across all 5 folds (see `python3
baseline/baseline_classifier.py` output). The methodology is now honest, but
this perfect score is itself a flag: the 400 labelled messages look
template-generated with very distinct per-class vocabulary (e.g. "2FA" /
"verification" for `account-access` vs. "phishing" / "drained" /
"unauthorized" for `fraud-report`). Real support traffic will be lexically
messier and more ambiguous (e.g. "someone hacked my account, can't log in"
could plausibly be either `fraud-report` or `account-access`), and this
dataset has no such boundary cases to test against. **A perfect CV score
here is not evidence the model will generalize to real traffic** — treat it
as an upper bound, not a production estimate.

**Metric to hold this service to in production:** `fraud-report` recall,
tracked on an ongoing basis against live-labelled traffic (not this static
CV number), because a false negative on fraud is the costliest error this
router can make. Report it alongside macro-F1 so the other three routes
aren't ignored. Accuracy alone should not be the number a PM signs off on.
