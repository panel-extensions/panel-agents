"""Wording of approval actions, shared by the policy and the approval components."""

from __future__ import annotations

APPROVAL_LABELS = ("approve", "deny", "approved", "denied", "edit", "approve_edited", "approved_edited", "approve_all", "deny_all")


def _past(verb: str) -> str:
    head, _, rest = verb.partition(" ")
    if head.endswith("e"):
        head += "d"
    elif len(head) > 1 and head.endswith("y") and head[-2].lower() not in "aeiou":
        head = head[:-1] + "ied"
    else:
        head += "ed"
    return f"{head} {rest}" if rest else head


def _gerund(verb: str) -> str:
    head, _, rest = verb.partition(" ")
    head = (head[:-1] if head.endswith("e") and not head.endswith("ee") else head) + "ing"
    return f"{head} {rest}" if rest else head


def _lower(text: str) -> str:
    return text[:1].lower() + text[1:]


def check_labels(labels: dict[str, str] | None) -> dict[str, str] | None:
    """Validate a labels mapping, raising ValueError for unknown keys or non-string values."""
    if labels is None:
        return None
    unknown = set(labels) - set(APPROVAL_LABELS)
    if unknown:
        raise ValueError(f"Unknown approval label(s) {sorted(unknown)}; expected some of {list(APPROVAL_LABELS)}.")
    if not all(isinstance(v, str) and v.strip() for v in labels.values()):
        raise ValueError("Approval labels must be non-empty strings.")
    return dict(labels)


def approval_labels(*overrides: dict[str, str] | None) -> dict[str, str]:
    """
    The wording of an approval's actions and outcomes, with later
    ``overrides`` taking precedence.

    Only ``approve`` and ``deny`` usually need to be given: ``approved``
    and ``denied`` are derived from them by adding "d" or "ed" (give them
    explicitly for irregular verbs), and ``approve_edited``,
    ``approved_edited``, ``approve_all`` and ``deny_all`` from those.
    Unknown keys are ignored, so recordings with newer labels still load.
    """
    given: dict[str, str] = {}
    for layer in overrides:
        given.update({k: v for k, v in (layer or {}).items() if k in APPROVAL_LABELS and isinstance(v, str) and v})
    approve, deny = given.get("approve", "Approve"), given.get("deny", "Deny")
    approved = given.get("approved", _past(approve))
    denied = given.get("denied", _past(deny))
    labels = {
        "approve": approve, "deny": deny, "approved": approved, "denied": denied, "edit": "Edit",
        "approve_edited": f"{approve} with edits", "approved_edited": f"{approved} with edits",
        "approve_all": f"{approve} all", "deny_all": f"{deny} all",
    }
    labels.update(given)
    return labels
