"""Apply only clearly supported review changes to the AI-assisted Golden Set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABELED_PATH = Path("evaluation/amazon_golden_set_labeled.csv")
REVIEW_PATH = Path("evaluation/golden_set_review.csv")
OUTPUT_PATH = Path("evaluation/amazon_golden_set_final.csv")

VALID_INTENTS = {
    "DELIVERY_ISSUE",
    "ORDER_ISSUE",
    "RETURN_REFUND",
    "PAYMENT_BILLING",
    "ACCOUNT_MEMBERSHIP",
    "PRODUCT_TECHNICAL",
    "PRODUCT_QUALITY",
    "SAFETY_ISSUE",
    "GENERAL_SUPPORT",
}
VALID_ACTIONS = {"AUTO_HANDLE", "ESCALATE"}
REQUIRED_COLUMNS = ["customer_message", "amazon_reply", "intent", "expected_action", "reason"]

# These review rows have a clearly supported correction. Review rows omitted
# here remain unchanged rather than being guessed at.
ACCEPTED_REVIEW_ROWS = {
    5, 6, 8, 19, 24, 26, 28, 29, 30, 33, 35, 37, 40, 50, 54, 59, 69, 73,
    74, 80, 86, 89, 90, 91, 97, 98, 99, 100, 103, 104, 105, 110, 111, 113,
    115, 116, 118, 119, 124, 126, 127, 131, 132, 141, 143, 145, 147, 151,
    154, 156, 160, 161, 164, 170, 174, 175, 176, 178, 179, 182, 183, 184,
    185, 186, 187, 188, 190, 191, 192, 193, 195, 198, 199,
}


def create_final_golden_set(
    labeled_path: str | Path = LABELED_PATH,
    review_path: str | Path = REVIEW_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> tuple[pd.DataFrame, int, int, int]:
    labeled = pd.read_csv(labeled_path, keep_default_na=False)
    review = pd.read_csv(review_path, keep_default_na=False)

    if list(labeled.columns) != REQUIRED_COLUMNS or len(labeled) != 200:
        raise ValueError("The labeled Golden Set must contain 200 rows and the required columns")
    if "row_number" not in review.columns:
        raise ValueError("The review report must contain row_number")

    final = labeled.copy()
    intent_changes = 0
    action_changes = 0
    changed_rows = set()

    for _, review_row in review.iterrows():
        row_number = int(review_row["row_number"])
        if row_number not in ACCEPTED_REVIEW_ROWS:
            continue

        row_index = row_number - 1
        suggested_intent = str(review_row["suggested_intent"])
        suggested_action = str(review_row["suggested_action"])
        if suggested_intent not in VALID_INTENTS or suggested_action not in VALID_ACTIONS:
            raise ValueError(f"Invalid suggestion at row {row_number}")

        if final.at[row_index, "intent"] != suggested_intent:
            intent_changes += 1
            changed_rows.add(row_number)
        if final.at[row_index, "expected_action"] != suggested_action:
            action_changes += 1
            changed_rows.add(row_number)
        final.at[row_index, "intent"] = suggested_intent
        final.at[row_index, "expected_action"] = suggested_action
        final.at[row_index, "reason"] = str(review_row["reason_for_review"])

    final.to_csv(output_path, index=False)
    unchanged = len(final) - len(changed_rows)
    return final, intent_changes, action_changes, unchanged


def main() -> None:
    final, intent_changes, action_changes, unchanged = create_final_golden_set()
    print(f"Rows written: {len(final)}")
    print("Final intent distribution:")
    print(final["intent"].value_counts().sort_index().to_string())
    print("Final action distribution:")
    print(final["expected_action"].value_counts().sort_index().to_string())
    print(f"Intent labels changed: {intent_changes}")
    print(f"Action labels changed: {action_changes}")
    print(f"Rows unchanged: {unchanged}")
    print("Provenance: AI-assisted draft requiring human review; not manually labelled.")


if __name__ == "__main__":
    main()