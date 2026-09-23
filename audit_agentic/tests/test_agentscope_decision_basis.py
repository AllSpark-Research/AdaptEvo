from audit_agentic.structured_output import normalize_decision_basis


def test_normalize_decision_basis_accepts_string_without_character_split():
    value = "1. 证据一\n2. 证据二"
    assert normalize_decision_basis(value) == [value]


def test_normalize_decision_basis_keeps_legacy_list_compatible():
    assert normalize_decision_basis(["证据一", " 证据二 ", ""]) == [
        "证据一",
        "证据二",
    ]
