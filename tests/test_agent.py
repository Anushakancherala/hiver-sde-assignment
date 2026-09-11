"""Tests for grounded reply generation without making real API calls."""

from src.agent import ESCALATION_RECOMMENDATION, generate_reply


class FakeProvider:
    def __init__(self, response: str = "Thanks for contacting AmazonHelp."):
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_generate_reply_uses_mocked_provider_and_all_grounding_inputs():
    provider = FakeProvider("Please check the delivery details and contact us for further help.")
    examples = [
        {
            "customer_message": "My package has not arrived.",
            "amazon_reply": "Please contact us so we can take a closer look at the delivery.",
            "similarity": 0.8,
        }
    ]

    reply = generate_reply("Where is my package?", "DELIVERY_ISSUE", examples, provider)

    assert reply == provider.response
    assert "Where is my package?" in provider.prompt
    assert "DELIVERY_ISSUE" in provider.prompt
    assert examples[0]["customer_message"] in provider.prompt
    assert examples[0]["amazon_reply"] in provider.prompt


def test_generate_reply_escalates_when_evidence_is_insufficient():
    provider = FakeProvider()

    reply = generate_reply(
        "I need help with something new.",
        "GENERAL_SUPPORT",
        [{"customer_message": "Old message", "amazon_reply": "Old reply", "similarity": 0.0}],
        provider,
    )

    assert reply == ESCALATION_RECOMMENDATION
    assert provider.prompt == ""