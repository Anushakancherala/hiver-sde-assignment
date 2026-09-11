"""Create a review report for obvious Golden Set label inconsistencies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("evaluation/amazon_golden_set_labeled.csv")
OUTPUT_PATH = Path("evaluation/golden_set_review.csv")

# Row numbers are 1-based and refer to the labeled Golden Set. These entries
# are limited to clear intent mismatches or action decisions needing review.
REVIEW_OVERRIDES = {
    5: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The message explicitly says two Prime deliveries never arrived."),
    6: ("PRODUCT_TECHNICAL", "AUTO_HANDLE", "The customer asks about an Echo/device's power, battery, and calling behavior."),
    8: ("ORDER_ISSUE", "ESCALATE", "The customer cannot cancel an order; cancellation handling may require human intervention."),
    19: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The message reports a changed promised delivery date."),
    24: ("DELIVERY_ISSUE", "ESCALATE", "The customer disputes a delivered status and says the package was not received."),
    26: ("DELIVERY_ISSUE", "ESCALATE", "The message describes a repeatedly delayed or diverted delivery."),
    28: ("PRODUCT_TECHNICAL", "ESCALATE", "The customer reports missing beta and digital content, requiring product/service investigation."),
    29: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The message concerns shipment tracking with Royal Mail."),
    30: ("ORDER_ISSUE", "AUTO_HANDLE", "The customer says the listed product cannot be ordered for an Indian pincode."),
    33: ("PRODUCT_QUALITY", "ESCALATE", "The customer received a visibly damaged package and faces a replacement problem."),
    35: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The customer stayed home for a delivery that is now delayed until tomorrow."),
    37: ("PRODUCT_TECHNICAL", "AUTO_HANDLE", "The message asks about an Echo device's technical capabilities."),
    40: ("RETURN_REFUND", "ESCALATE", "A returned product is disputed after a month, requiring human review."),
    50: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The customer reports that the delivery is delayed until tomorrow."),
    54: ("ACCOUNT_MEMBERSHIP", "ESCALATE", "The message combines connection and incorrect-password/account access problems."),
    59: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The message asks whether an order's promised delivery meets Prime delivery expectations."),
    69: ("RETURN_REFUND", "ESCALATE", "The customer says a refund remains missing despite repeated apologies."),
    73: ("GENERAL_SUPPORT", "AUTO_HANDLE", "The customer requests a direct support phone number, not account help."),
    74: ("PAYMENT_BILLING", "ESCALATE", "The bank cannot find the Amazon transaction, so the payment discrepancy needs review."),
    80: ("RETURN_REFUND", "ESCALATE", "The customer is still waiting for a refund email."),
    86: ("GENERAL_SUPPORT", "ESCALATE", "The unresolved issue concerns repeated failed contact and a missed callback."),
    89: ("DELIVERY_ISSUE", "ESCALATE", "The customer asks about accommodating delivery-driver activity at a residence."),
    90: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The customer's Halloween costume delivery is one day late."),
    91: ("GENERAL_SUPPORT", "ESCALATE", "The customer cannot find the correspondence needed to continue support."),
    97: ("GENERAL_SUPPORT", "ESCALATE", "Multiple calls produced no resolution or promised callback."),
    98: ("ORDER_ISSUE", "AUTO_HANDLE", "The message questions a Prime promise shown during ordering."),
    99: ("GENERAL_SUPPORT", "ESCALATE", "The customer has no available support call option."),
    100: ("GENERAL_SUPPORT", "ESCALATE", "Customer care disconnected the customer and refused further help."),
    103: ("GENERAL_SUPPORT", "ESCALATE", "The customer cannot contact Amazon because calls are blocked and the issue is unresolved."),
    104: ("ACCOUNT_MEMBERSHIP", "ESCALATE", "The message concerns a discontinued subscription/service and requires account support."),
    105: ("DELIVERY_ISSUE", "ESCALATE", "The customer's Prime delivery failed for two days and caused a serious complaint."),
    107: ("GENERAL_SUPPORT", "ESCALATE", "The customer reports repeated lack of updates from support."),
    110: ("DELIVERY_ISSUE", "ESCALATE", "The order was undeliverable and the customer asks about receiving a refund."),
    111: ("GENERAL_SUPPORT", "ESCALATE", "The customer is receiving neither email nor phone support and threatens to leave."),
    113: ("RETURN_REFUND", "ESCALATE", "The customer requests compensation and a full refund, requiring human review."),
    115: ("PAYMENT_BILLING", "ESCALATE", "The displayed price changes dramatically at card selection, indicating a billing risk."),
    116: ("PAYMENT_BILLING", "ESCALATE", "The customer cannot use a gift-card payment code to complete a purchase."),
    118: ("RETURN_REFUND", "ESCALATE", "A promised refund has not been completed despite multiple calls."),
    119: ("ACCOUNT_MEMBERSHIP", "ESCALATE", "The customer reports being scammed and wants to cancel a membership."),
    124: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The customer reports that a delivery attempt was never made."),
    126: ("GENERAL_SUPPORT", "ESCALATE", "The customer wants to know what action was taken on a complaint."),
    127: ("DELIVERY_ISSUE", "ESCALATE", "The tracking status contradicts the failed-delivery explanation."),
    131: ("GENERAL_SUPPORT", "ESCALATE", "The customer reports repeated unresolved service failure and requests compensation."),
    132: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The customer is discussing carrier contact and delivery information."),
    141: ("DELIVERY_ISSUE", "AUTO_HANDLE", "The message concerns courier pickup scheduling and the time required at home."),
    143: ("RETURN_REFUND", "ESCALATE", "The refund issue remains unresolved and the seller is not responding."),
    145: ("RETURN_REFUND", "ESCALATE", "The message explicitly says the refund process has not completed."),
    147: ("ORDER_ISSUE", "ESCALATE", "The row contains only an order-details link, so there is insufficient issue detail for auto-handling."),
    151: ("ACCOUNT_MEMBERSHIP", "ESCALATE", "The customer cannot log in and therefore cannot complete the support page."),
    154: ("PAYMENT_BILLING", "ESCALATE", "The customer was charged after being told membership would not renew."),
    156: ("DELIVERY_ISSUE", "ESCALATE", "The customer disputes a delivered scan and reports a repeated non-delivery."),
    160: ("DELIVERY_ISSUE", "ESCALATE", "The primary issue is a Prime shipment more than a week late; the requested refund adds risk."),
    161: ("PRODUCT_TECHNICAL", "AUTO_HANDLE", "The customer reports missing episodes from a digital viewing service."),
    164: ("GENERAL_SUPPORT", "AUTO_HANDLE", "The customer confirms the package arrived and expresses thanks; escalation is unnecessary."),
    166: ("DELIVERY_ISSUE", "ESCALATE", "The customer says something was not received and asks for a resend, but details are incomplete."),
    167: ("ORDER_ISSUE", "ESCALATE", "The customer raises several order/catalog/Prime questions with no single safe resolution."),
    170: ("ACCOUNT_MEMBERSHIP", "ESCALATE", "The customer requests account closure after unresolved support contact."),
    174: ("GENERAL_SUPPORT", "ESCALATE", "The customer requests a call to resolve an unanswered support matter."),
    175: ("GENERAL_SUPPORT", "ESCALATE", "The customer reports that support operators are not processing the email."),
    176: ("ORDER_ISSUE", "ESCALATE", "The ordered router was returned to fulfillment without permission, requiring investigation."),
    178: ("GENERAL_SUPPORT", "ESCALATE", "Repeated reports and customer-service contact have produced no help."),
    179: ("DELIVERY_ISSUE", "ESCALATE", "The customer reports both missing money/package and requests an escalation contact."),
    182: ("ORDER_ISSUE", "ESCALATE", "The customer cannot cancel an order and needs order-level intervention."),
    183: ("DELIVERY_ISSUE", "ESCALATE", "The delivery scan says handed directly to the customer, but the customer was absent."),
    184: ("PRODUCT_QUALITY", "ESCALATE", "The customer reports a product warranty dispute after a long unresolved repair issue."),
    185: ("PAYMENT_BILLING", "ESCALATE", "The customer disputes being charged more than the stated maximum retail price."),
    186: ("RETURN_REFUND", "ESCALATE", "The customer disputes pickup charges while waiting for a refund."),
    187: ("GENERAL_SUPPORT", "ESCALATE", "The customer waited 15 days and was told support cannot help despite escalation."),
    188: ("PAYMENT_BILLING", "ESCALATE", "The message concerns a charge/debit and changing or cancelling a paid membership."),
    190: ("GENERAL_SUPPORT", "ESCALATE", "The customer has sent many emails without a useful response and requests a call."),
    191: ("PRODUCT_TECHNICAL", "ESCALATE", "Fire Tablet malfunction and a warranty dispute are technical and unresolved; 'fire' is the product name here."),
    192: ("GENERAL_SUPPORT", "ESCALATE", "The customer reports an incorrect support answer and needs human clarification."),
    193: ("GENERAL_SUPPORT", "AUTO_HANDLE", "This is a thank-you message with no unresolved issue; escalation is unnecessary."),
    195: ("PAYMENT_BILLING", "ESCALATE", "A promised cashback payment missed its deadline."),
    198: ("ORDER_ISSUE", "ESCALATE", "The customer was told to cancel but the order remains active, requiring intervention."),
    199: ("DELIVERY_ISSUE", "ESCALATE", "The customer disputes a delivered status and says they were home but received nothing."),
}


def create_review_report(input_path: str | Path = INPUT_PATH, output_path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    labeled = pd.read_csv(input_path, keep_default_na=False)
    rows = []
    for row_number, (suggested_intent, suggested_action, reason) in REVIEW_OVERRIDES.items():
        row = labeled.iloc[row_number - 1]
        rows.append(
            {
                "row_number": row_number,
                "customer_message": row["customer_message"],
                "current_intent": row["intent"],
                "suggested_intent": suggested_intent,
                "current_action": row["expected_action"],
                "suggested_action": suggested_action,
                "reason_for_review": reason,
            }
        )
    report = pd.DataFrame(
        rows,
        columns=[
            "row_number",
            "customer_message",
            "current_intent",
            "suggested_intent",
            "current_action",
            "suggested_action",
            "reason_for_review",
        ],
    )
    report.to_csv(output_path, index=False)
    return report


if __name__ == "__main__":
    report = create_review_report()
    intent_issues = int((report["current_intent"] != report["suggested_intent"]).sum())
    action_issues = int((report["current_action"] != report["suggested_action"]).sum())
    print(f"Total rows reviewed: 200")
    print(f"Potentially incorrect/ambiguous rows: {len(report)}")
    print(f"Intent issues: {intent_issues}")
    print(f"Action issues: {action_issues}")