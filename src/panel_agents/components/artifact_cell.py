"""
An artifact shown inline in a feed: its kind, title, actions and value.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..actions import ActionRegistry
from ..artifacts.model import Artifact
from ..security import _escape_markdown, _plain_typography, untrusted_markdown
from .artifacts import _kind_icon
from .renderers import render

logger = logging.getLogger(__name__)

_CELL_CSS = """
:host {
  border: 1px solid var(--mui-palette-divider);
  border-radius: var(--mui-shape-borderRadius, 8px);
  background: var(--mui-palette-background-paper);
  overflow: hidden;
}
"""

_HEADER_CSS = """
:host { border-bottom: 1px solid var(--mui-palette-divider); }
"""


def _title_sx(title: str) -> dict[str, str]:
    """A name such as a table or file name is set in monospace; a title in words is not."""
    sx = {"overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}
    if title and not any(ch.isspace() for ch in title):
        sx["fontFamily"] = "monospace"
    return sx


@dataclass(frozen=True)
class ArtifactAction:
    """
    A button offered on artifact cells that runs an action of an
    ``ActionRegistry``, e.g. "Add to dashboard".

    ``arguments`` maps an artifact to the action's arguments, or to None
    when the action does not apply to it, which hides the button. ``done``
    maps an artifact to a short label shown instead of the button once
    the action has nothing left to do for it, e.g. "On dashboard", or to
    None.
    """

    action: str
    label: str
    icon: str | None = None
    arguments: Callable[[Artifact], dict[str, Any] | None] = lambda artifact: {}
    done: Callable[[Artifact], str | None] | None = None


class ArtifactCell(Viewer):
    """
    A card showing an artifact's latest version inline, under a header
    with its kind, its title and the actions that apply to it.

    The value is rendered by ``renderer`` or the renderer registry, and
    rendered again when a new version is published.
    """

    artifact = param.ClassSelector(class_=Artifact, constant=True)

    renderer = param.Callable(default=None, allow_None=True, doc="""
        Callable ``(artifact) -> Viewable`` rendering the cell's body, e.g.
        a host's view of the live object an artifact refers to. Defaults
        to ``render(value, kind)``.""")

    actions = param.ClassSelector(class_=ActionRegistry, default=None, allow_None=True, doc="""
        The registry that runs the cell's actions.""")

    artifact_actions = param.List(default=[], item_type=ArtifactAction, doc="""
        The actions offered on the cell; those whose action the registry
        does not have are left out.""")

    def __init__(self, artifact: Artifact | None = None, *, styles: dict | None = None, **params):
        if artifact is not None:
            params["artifact"] = artifact
        super().__init__(**params)
        artifact = self.artifact
        self._chip = pmui.Chip(
            label=artifact.kind, icon=_kind_icon(artifact.kind), size="small",
            variant="outlined", margin=(0, 8, 0, 0), css_classes=["artifact-cell-kind"],
        )
        # Titles and errors may carry injected content, so they are plain text.
        title = artifact.title or artifact.artifact_id
        self._title = _plain_typography(
            title, variant="subtitle2", margin=0, sx=_title_sx(title),
            styles={"flex": "1 1 120px", "min-width": "0"}, css_classes=["artifact-cell-title"],
        )
        self._actions = pn.Row(margin=0, styles={"margin-left": "auto"}, css_classes=["artifact-cell-actions"])
        self._error = _plain_typography(
            "", variant="caption", visible=False, margin=(4, 12),
            sx={"color": "error.main"}, css_classes=["artifact-cell-error"],
        )
        self._body = pn.Column(sizing_mode="stretch_width", margin=0, css_classes=["artifact-cell-body"])
        # On a narrow card the actions wrap below the title rather than being cut off.
        header = pn.FlexBox(
            self._chip, self._title, self._actions,
            flex_wrap="wrap", align_items="center", gap="4px 0",
            sizing_mode="stretch_width", margin=0, styles={"padding": "6px 12px"},
            stylesheets=[_HEADER_CSS], css_classes=["artifact-cell-header"],
        )
        self._layout = pn.Column(
            header, self._error, self._body, sizing_mode="stretch_width", margin=0,
            stylesheets=[_CELL_CSS], css_classes=["artifact-cell"], styles=styles or {},
        )
        self._rendered: int | None = None
        self._controls: dict[tuple, Any] = {}
        self._watchers = [
            artifact.param.watch(self._update_title, "title"),
            artifact.param.watch(self._update_body, "versions"),
        ]
        self._update_body()
        self.refresh()

    def __panel__(self):
        return self._layout

    def release(self):
        """Stop following the artifact, e.g. when the feed showing the cell is cleared."""
        for watcher in self._watchers:
            self.artifact.param.unwatch(watcher)
        self._watchers = []

    def _update_title(self, *_):
        title = self.artifact.title or self.artifact.artifact_id
        self._title.param.update(object=_escape_markdown(title), sx=_title_sx(title))

    def _update_body(self, *_):
        latest = self.artifact.latest
        number = latest.version if latest is not None else None
        if number == self._rendered:
            return
        self._rendered = number
        self._body.objects = [self._render()]

    def _render(self):
        artifact = self.artifact
        try:
            if self.renderer is None:
                return render(artifact.value, artifact.kind)
            view = self.renderer(artifact)
        except Exception as exc:
            # A broken renderer must not stop the run that published the artifact.
            logger.exception("Rendering the artifact %s failed", artifact.artifact_id)
            return _plain_typography(
                f"This artifact could not be shown: {exc}", variant="body2", margin=(8, 12),
                sx={"color": "error.main"}, css_classes=["artifact-cell-render-error"],
            )
        if isinstance(view, str):
            return untrusted_markdown(view, sizing_mode="stretch_width")
        return view

    @param.depends("renderer", watch=True)
    def _rerender(self):
        self._rendered = None
        self._update_body()

    @param.depends("actions", "artifact_actions", watch=True)
    def refresh(self):
        """Evaluate again which actions apply to the artifact and which are done."""
        objects, controls = [], {}
        for spec in self.artifact_actions:
            if self.actions is None or spec.action not in self.actions:
                continue
            label = spec.done(self.artifact) if spec.done is not None else None
            if not label and spec.arguments(self.artifact) is None:
                continue
            key = (spec, label)
            # A control whose state is unchanged is kept, so a click on it is never lost to a refresh.
            control = self._controls.get(key) or self._control(spec, label)
            controls[key] = control
            objects.append(control)
        self._controls = controls
        if [id(o) for o in self._actions.objects] != [id(o) for o in objects]:
            self._actions.objects = objects

    def _control(self, spec: ArtifactAction, label: str | None):
        if label:
            return pmui.Chip(
                label=label, icon="check", size="small", color="success", variant="outlined",
                margin=0, css_classes=["artifact-cell-done"],
            )
        button = pmui.Button(
            label=spec.label, icon=spec.icon, variant="outlined", size="small", margin=0,
            sx={"textTransform": "none"}, css_classes=["artifact-cell-action"],
        )
        button.on_click(lambda _: pn.state.execute(partial(self._invoke, spec, button)))
        return button

    async def _invoke(self, spec: ArtifactAction, button: pmui.Button):
        """Run ``spec`` on the artifact as it is now, with the arguments it gives at the click."""
        button.loading = True
        self._error.visible = False
        try:
            arguments = spec.arguments(self.artifact)
            if arguments is None:
                raise ValueError("it no longer applies")
            result = self.actions.invoke(spec.action, arguments)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.info("The %r action failed on %s: %s", spec.action, self.artifact.artifact_id, exc)
            self._error.param.update(object=_escape_markdown(f"{spec.label} failed: {exc}"), visible=True)
        finally:
            button.loading = False
        self.refresh()


__all__ = ["ArtifactAction", "ArtifactCell"]
