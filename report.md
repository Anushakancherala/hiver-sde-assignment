# Hiver SDE Intern Assignment Report

## Problem framing

The task is to classify AmazonHelp-style customer messages into nine intent categories and decide whether each request should be auto-handled or escalated. The repo evaluates on the final reviewed Golden Set of 200 rows, and the result files explicitly say the Golden Set was not used for training (`golden_set_used_for_training: false`).

The challenge is not just classification accuracy. The label distribution is highly skewed: in the final Golden Set, `GENERAL_SUPPORT` has 71 examples, while several other labels have small support counts, including `SAFETY_ISSUE` (1), `PRODUCT_QUALITY` (3), and `ORDER_ISSUE` (9). That imbalance makes aggregate accuracy easy to over-interpret and makes minority but high-risk cases especially important.

## System approach

The implementation is a simple and explainable pipeline:

- a rule-assisted historical training dataset is used to build the intent classifier,
- the classifier is TF-IDF + logistic regression,
- a retrieval layer looks up similar historical conversation pairs,
- a grounded reply generator uses retrieved examples and a conservative escalation policy,
- the final action is chosen with a rule-based `AUTO_HANDLE` vs `ESCALATE` decision.

This is intentionally lightweight and auditable. It does not claim any human-labeled benchmark beyond the final reviewed Golden Set, and the judge output is clearly marked as `mock_only_not_final_quality`.

## Comparison against trivial and simple baselines

The repo includes a trivial baseline and a simple heuristic baseline, both evaluated on the same final Golden Set.

Measured values from the files:

- `trivial_baseline`: intent accuracy = 0.355, intent macro F1 = 0.0582, action accuracy = 0.420, action macro F1 = 0.2958
- `simple_baseline`: intent accuracy = 0.405, intent macro F1 = 0.1507, action accuracy = 0.460, action macro F1 = 0.3689
- `existing_agent`: intent accuracy = 0.405, intent macro F1 = 0.1507, action accuracy = 0.435, action macro F1 = 0.3241

Key takeaways:

- The current pipeline beats the trivial baseline on both intent and action metrics.
- The current pipeline matches the simple baseline on intent accuracy and macro F1.
- The current pipeline is worse than the simple baseline on action accuracy and action macro F1.

This means the system is not clearly outperforming the low-complexity baseline on the final Golden Set; it is roughly competitive on intent and weaker on action selection.

## Top 5 failure modes with concrete examples and hypotheses

### 1) Majority-class collapse on minority intents

The confusion matrix shows a consistent pattern: many examples are predicted as `GENERAL_SUPPORT` instead of their true minority labels. The final evaluation file shows `DELIVERY_ISSUE` (40 examples) being collapsed to `GENERAL_SUPPORT` in 38 cases, with similar behavior for `RETURN_REFUND` and `ACCOUNT_MEMBERSHIP`.

Example from the Golden Set:

- Message: `@AmazonHelp 1. Does it work without being plugged in to power? 2. If yes, what is battery life? 3. Can it be used for receiving and making calls?`
- Gold label: `PRODUCT_TECHNICAL`

Hypothesis: the model learns the dominant class pattern and underfits the minority intents because the training data is imbalanced and the messages are short and noisy.

### 2) Technical questions are not recognized

`PRODUCT_TECHNICAL` is a class with clear support in the Golden Set but is effectively missed by the model. The file shows this label has zero F1 in the final evaluation.

Example from the Golden Set:

- Message: `@AmazonHelp 1. Does it work without being plugged in to power? 2. If yes, what is battery life? 3. Can it be used for receiving and making calls?`
- Gold intent: `PRODUCT_TECHNICAL`
- Gold action: `AUTO_HANDLE`

Hypothesis: technical support terminology is sparse in the training data, so the model defaults to a generic support bucket instead of identifying a device-related technical issue.

### 3) Escalation decisions are too weak

The action metric shows poor escalation recall. For the existing agent, `ESCALATE` recall is only 0.0259, with action accuracy at 0.435.

Examples from the Golden Set:

- `@AmazonHelp i reset my password 3 times and it still says incorrect you gotta be shitting me` -> `ACCOUNT_MEMBERSHIP`, `ESCALATE`
- `@AmazonHelp Your customer service has reverted saying that apparently the package has been delivered and signed by receipt my yesterday. Not true.` -> `DELIVERY_ISSUE`, `ESCALATE`
- `@AmazonHelp there was fire on stove n if cylinder got fired what worst could happened...` -> `SAFETY_ISSUE`, `ESCALATE`

Hypothesis: the escalation rule is too conservative and misses many high-risk or frustrated customer states because they often appear in short, informal, and ambiguous social-media language rather than explicit safety keywords.

### 4) Payment, account, and refund confusion

The model struggles to separate semantically adjacent cases around money, refunds, and access issues. The final evaluation file shows multiple labels with zero or near-zero F1.

Examples from the Golden Set:

- `@AmazonHelp No use of prime membership, return my money` -> `PAYMENT_BILLING`, `AUTO_HANDLE`
- `@AmazonHelp I was told my account is on hold, if I can’t use my gift cards please return my money back to me Okay` -> `PAYMENT_BILLING`, `AUTO_HANDLE`
- `@AmazonHelp Not impressed with Amazon Logistics. Computer parts delayed a week and a refund is the only option?` -> `RETURN_REFUND`, `AUTO_HANDLE`

Hypothesis: the model relies on word overlap rather than deeper distinctions between payment, account access, and refund intent, causing these classes to get mixed together.

### 5) The generated reply is generic and not strongly evidence-bound

The judge results show a flat value of 3.0 for relevance, groundedness, correctness, helpfulness, and overall score across all 40 judged examples. This is not a strong quality signal; it is a mid-scale default pattern.

Concrete example in the artifact:

- Generated reply: `Based on similar AmazonHelp cases, please contact Amazon customer support for assistance.`

Hypothesis: the retrieval and reply generation path is too generic and does not anchor the response to the actual customer request or the retrieved evidence, producing a template-like output across many different issues.

## What is misleading about my headline number?

The headline number is the aggregate accuracy, but that number masks the actual failure pattern. For the current system, intent accuracy is 0.405 and action accuracy is 0.435, yet the macro F1 is far lower: intent macro F1 is 0.1507 and action macro F1 is 0.3241.

This is misleading because the class distribution is imbalanced. `GENERAL_SUPPORT` is the dominant class, and the system often predicts it. The confusion matrix makes that clear: the model is performing well on the majority bucket while missing minority labels and high-risk cases that matter operationally.

The mock judge file adds another warning: it is explicitly labeled `mode: "mock_only_not_final_quality"`, so it should not be interpreted as final quality evidence.

## One-week next steps

1. Rebalance the training data and add more examples for the minority labels, especially `PRODUCT_TECHNICAL`, `PAYMENT_BILLING`, `ORDER_ISSUE`, and `SAFETY_ISSUE`.
2. Rework the escalation rule to better detect genuine risk signals, account issues, and repeated unresolved complaint patterns.
3. Improve the retrieval layer so generated replies are tied to the actual customer issue rather than a generic fallback template.
4. Review per-label performance on the final Golden Set instead of relying only on aggregate accuracy.
5. Keep judge evaluation honest: treat the mock judge as a debug signal only unless a non-mock judge run is generated and recorded in the artifacts.

The current files support a clear conclusion: the system is a useful baseline, but it is not yet a robust production triage system on the final Golden Set.
