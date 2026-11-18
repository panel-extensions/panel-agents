"""
Declarative rules deciding which tool calls need human approval and who
may give it.

A policy is independent of any agent framework: an adapter asks
``requires_approval`` before a tool runs and ``can_decide`` before it
acts on a decision, and approval components use ``can_approve`` as their
permission hook.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._labels import FrozenLabels, check_labels
from .state import ApprovalRequest

_Approvers = Collection[str] | Callable[[ApprovalRequest, str | None], bool]


@dataclass(frozen=True)
class ApprovalRule:
    """
    Requires approval for calls to the tools matching ``tool``.

    Parameters
    ----------
    tool
        Tool name, or a glob pattern such as ``"delete_*"``.
    when
        Predicate on the call arguments, with defaults filled in; the rule
        only applies to calls for which it returns True. None applies it
        to every call. A predicate that raises counts as True, so a broken
        predicate asks for approval rather than letting the call through.
    approvers
        Who may decide: a collection of user names, or a callable
        ``(request, user) -> bool``. ``user`` is None when the app has no
        authentication, and a callable that raises denies. None lets
        anyone decide.
    labels
        Wording of the decision actions on the approval card, queue and
        tool-call badge, e.g. ``{"deny": "Remove", "denied": "Removed"}``
        for a plan step. It is recorded on the ``ApprovalRequested``
        event, so replayed sessions show the same words. See
        ``ApprovalCard.labels`` for the keys.
    """

    tool: str

    when: Callable[[dict[str, Any]], bool] | None = None

    approvers: _Approvers | None = None

    labels: Mapping[str, str] | None = None

    def __post_init__(self):
        if self.labels is not None:
            object.__setattr__(self, "labels", FrozenLabels(check_labels(self.labels)))

    def matches(self, tool_name: str, args: dict[str, Any] | None = None) -> bool:
        """Whether a call to ``tool_name`` with ``args`` needs approval under this rule."""
        if not fnmatch.fnmatchcase(tool_name, self.tool):
            return False
        if self.when is None:
            return True
        try:
            return bool(self.when(dict(args or {})))
        except Exception:
            return True

    def permits(self, request: ApprovalRequest, user: str | None) -> bool:
        """Whether ``user`` may approve or deny ``request``."""
        if self.approvers is None:
            return True
        if callable(self.approvers):
            try:
                return bool(self.approvers(request, user))
            except Exception:
                return False
        return user is not None and user in self.approvers


class ApprovalPolicy:
    """
    An ordered list of ``ApprovalRule``s; the first matching rule decides.

    Strings are shorthand for a rule requiring approval for every call to
    that tool. Calls matching no rule need no approval from the policy,
    though tools can still ask for it themselves.

    >>> policy = ApprovalPolicy([
    ...     "save_*",
    ...     ApprovalRule("run_sql", when=lambda args: not args["sql"].lower().startswith("select")),
    ...     ApprovalRule("publish", approvers={"admin"}),
    ... ])
    >>> policy.requires_approval("save_dashboard", {})
    True
    """

    def __init__(self, rules: Sequence[ApprovalRule | str] = ()):
        self.rules: list[ApprovalRule] = [ApprovalRule(rule) if isinstance(rule, str) else rule for rule in rules]

    def __repr__(self) -> str:
        return f"ApprovalPolicy({self.rules!r})"

    def rule_for(self, tool_name: str, args: dict[str, Any] | None = None) -> ApprovalRule | None:
        """The first rule matching the call, if any."""
        return next((rule for rule in self.rules if rule.matches(tool_name, args)), None)

    def requires_approval(self, tool_name: str, args: dict[str, Any] | None = None) -> bool:
        """Whether a call to ``tool_name`` with ``args`` must be approved before it runs."""
        return self.rule_for(tool_name, args) is not None

    def can_approve(self, request: ApprovalRequest, user: str | None) -> bool:
        """
        Whether ``user`` may decide ``request`` with its arguments; matches
        the ``can_approve`` hook of ``ApprovalCard`` and ``ApprovalQueue``.

        Requests no rule matches, e.g. from tools asking for approval
        themselves, can be decided by anyone.
        """
        rule = self.rule_for(request.tool_name, request.args)
        return rule is None or rule.permits(request, user)

    def can_decide(self, request: ApprovalRequest, user: str | None, args: dict[str, Any] | None = None) -> bool:
        """
        Whether ``user`` may approve ``request`` with ``args``, the final
        arguments after any edits. Both the proposed and the final
        arguments must allow the user, so editing a call cannot move it
        out of the user's reach.
        """
        if not self.can_approve(request, user):
            return False
        return args is None or self.can_approve(_with_args(request, args), user)


def _with_args(request: ApprovalRequest, args: dict[str, Any]) -> ApprovalRequest:
    return ApprovalRequest(
        tool_call_id=request.tool_call_id, tool_name=request.tool_name, args=dict(args),
        schema=request.schema, description=request.description, run_id=request.run_id,
        labels=request.labels,
    )


__all__ = ["ApprovalPolicy", "ApprovalRule"]
