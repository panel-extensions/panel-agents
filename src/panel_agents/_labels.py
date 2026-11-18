"""Wording of approval actions, shared by the policy and the approval components."""

from __future__ import annotations

from collections.abc import Mapping

import param

APPROVAL_LABELS = ("approve", "deny", "approved", "denied", "edit", "approve_edited", "approved_edited", "approve_all", "deny_all")

_DEFAULTS = {
    "approve": "Approve", "deny": "Deny", "approved": "Approved", "denied": "Denied", "edit": "Edit",
    "approve_edited": "Approve with edits", "approved_edited": "Approved with edits",
    "approve_all": "Approve all", "deny_all": "Deny all",
}

# A changed verb needs its past tense, which cannot be derived reliably.
_PAST = {"approve": "approved", "deny": "denied"}


def _lower(text: str) -> str:
    return text[:1].lower() + text[1:]


def _missing_past(labels: Mapping[str, str]) -> list[str]:
    return [
        past for verb, past in _PAST.items()
        if verb in labels and labels[verb] != _DEFAULTS[verb] and past not in labels
    ]


def check_labels(labels: Mapping[str, str] | None) -> dict[str, str] | None:
    """
    Validate a labels mapping, raising ValueError for unknown keys,
    empty or non-string values, and a changed ``approve`` or ``deny``
    without its past tense.
    """
    if labels is None:
        return None
    unknown = set(labels) - set(APPROVAL_LABELS)
    if unknown:
        raise ValueError(f"Unknown approval label(s) {sorted(unknown)}; expected some of {list(APPROVAL_LABELS)}.")
    if not all(isinstance(v, str) and v.strip() for v in labels.values()):
        raise ValueError("Approval labels must be non-empty strings.")
    if missing := _missing_past(labels):
        given = ", ".join(f"{verb}={labels[verb]!r}" for verb, past in _PAST.items() if past in missing)
        raise ValueError(f"{given} also needs {' and '.join(repr(m) for m in missing)}, the wording once decided, e.g. {{'deny': 'Remove', 'denied': 'Removed'}}.")
    return dict(labels)


def _apply(labels: dict[str, str], layer: Mapping[str, str]) -> dict[str, str]:
    layer = {k: v for k, v in layer.items() if k in APPROVAL_LABELS and isinstance(v, str) and v.strip()}
    # A recorded layer that fails validation keeps the verbs of the layers below it.
    for verb in _missing_past(layer):
        layer.pop(next(v for v, p in _PAST.items() if p == verb))
    out = dict(labels)
    if "approve" in layer or "approved" in layer:
        approve = layer.get("approve", out["approve"])
        approved = layer.get("approved", _DEFAULTS["approved"] if approve == _DEFAULTS["approve"] else out["approved"])
        out.update(
            approve=approve, approved=approved, approve_all=f"{approve} all",
            approve_edited=f"{approve} with edits", approved_edited=f"{approved} with edits",
        )
    if "deny" in layer or "denied" in layer:
        deny = layer.get("deny", out["deny"])
        denied = layer.get("denied", _DEFAULTS["denied"] if deny == _DEFAULTS["deny"] else out["denied"])
        out.update(deny=deny, denied=denied, deny_all=f"{deny} all")
    out.update(layer)
    return out


def approval_labels(*layers: Mapping[str, str] | None) -> dict[str, str]:
    """
    The wording of an approval's actions and outcomes, resolving each
    layer in turn over the defaults.

    A layer that sets ``approve`` or ``deny`` also resets the wording
    derived from it in earlier layers: ``approve_all``, ``deny_all``,
    ``approve_edited`` and ``approved_edited``. A changed verb must come
    with its past tense (``approved``, ``denied``); a layer from a
    recording that lacks it keeps the earlier verb. Unknown keys are
    ignored, so recordings with newer labels still load.
    """
    labels = dict(_DEFAULTS)
    for layer in layers:
        if layer:
            labels = _apply(labels, layer)
    return labels


def _uses_default_verbs(labels: Mapping[str, str]) -> bool:
    return labels["approve"] == _DEFAULTS["approve"] and labels["deny"] == _DEFAULTS["deny"]


class Labels(param.Dict):
    """A ``param.Dict`` of approval labels, validated with ``check_labels``."""

    def _validate_value(self, value, allow_None):
        super()._validate_value(value, allow_None)
        check_labels(value)
