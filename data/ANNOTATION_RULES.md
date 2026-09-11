# AmazonHelp Rule-Assisted Annotation

`annotate_training_data.py` creates `data/amazonhelp_training_labeled.csv`
from `data/amazonhelp_conversation_pairs.csv`. It reads only the historical
conversation-pair CSV. It does not read, change, or use the Golden Set.

Run it from the repository root:

```text
python -m src.intents.annotate_training_data
```

The output contains:

- `customer_message`: the original customer message.
- `intent`: one of the nine intents defined in `src/intents/classifier.py`.
- `label_source`: always `rule_assisted`.
- `confidence`: `high`, `medium`, or `low`.

## Rules

The editable rules are in `src/intents/annotate_training_data.py`. They are
case-insensitive regular expressions grouped by intent. Examples include:

| Intent | Example signals |
| --- | --- |
| `DELIVERY_ISSUE` | delivery, package, tracking, late, not arrived |
| `ORDER_ISSUE` | order, purchase, cancel order, pre-order, out of stock |
| `RETURN_REFUND` | return, refund, money back, exchange |
| `PAYMENT_BILLING` | payment, billing, charged, credit card, invoice |
| `ACCOUNT_MEMBERSHIP` | account, login, password, Prime, membership |
| `PRODUCT_TECHNICAL` | Echo, Alexa, device, app, Wi-Fi, not working |
| `PRODUCT_QUALITY` | broken, defective, damaged, faulty, wrong item |
| `SAFETY_ISSUE` | unsafe, dangerous, fire, smoke, injury, hazard |
| `GENERAL_SUPPORT` | help, support, information, thanks, greeting |

Each matching rule contributes a score. The highest-scoring intent is chosen.
An unmatched message or a tie is assigned `GENERAL_SUPPORT` with `low`
confidence. A strong, clearly separated score is `high`; other non-tied
matches are `medium`. Low-confidence rows must be reviewed before treating
them as dependable training labels.

These are automatically generated heuristic labels, not human annotations.
The output is therefore a candidate training dataset and should be audited
before classifier training.