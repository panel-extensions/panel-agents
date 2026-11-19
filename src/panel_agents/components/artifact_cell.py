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
from .renderers import render

logger = logging.getLogger(__name__)

_KIND_ICONS = {
    "table": "table_chart",
    "sql": "code",
    "code": "code",
    "python": "code",
    "json": "data_object",
    "text": "article",
    "markdown": "article",
    "plot": "insert_chart",
    "chart": "insert_chart",
}

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

    def __init__(self, artifact: Artifact | None = None, **params):
        if artifact is not None:
            params["artifact"] = artifact
        super().__init__(**params)
        artifact = self.artifact
        self._chip = pmui.Chip(
            label=artifact.kind, icon=_KIND_ICONS.get(artifact.kind, "inventory_2"), size="small",
            variant="outlined", margin=(0, 8, 0, 0), css_classes=["artifact-cell-kind"],
        )
        self._title = pmui.Typography(
            artifact.title or artifact.artifact_id, variant="subtitle2", margin=0,
            sx={"fontFamily": "monospace", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"},
            css_classes=["artifact-cell-title"],
        )
        self._actions = pn.Row(margin=0, css_classes=["artifact-cell-actions"])
        self._error = pmui.Typography(
            "", variant="caption", visible=False, margin=(4, 12),
            sx={"color": "error.main"}, css_classes=["artifact-cell-error"],
        )
        self._body = pn.Column(sizing_mode="stretch_width", margin=0, css_classes=["artifact-cell-body"])
        header = pn.Row(
            self._chip, self._title, pn.HSpacer(), self._actions,
            sizing_mode="stretch_width", margin=0, styles={"align-items": "center", "padding": "6px 12px"},
            stylesheets=[_HEADER_CSS], css_classes=["artifact-cell-header"],
        )
        self._layout = pn.Column(
            header, self._error, self._body, sizing_mode="stretch_width", margin=0,
            stylesheets=[_CELL_CSS], css_classes=["artifact-cell"],
        )
        self._rendered: int | None = None
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
        self._title.object = self.artifact.title or self.artifact.artifact_id

    def _update_body(self, *_):
        latest = self.artifact.latest
        number = latest.version if latest is not None else None
        if number == self._rendered:
            return
        self._rendered = number
        self._body.objects = [self._render()]

    def _render(self):
        artifact = self.artifact
        if self.renderer is not None:
            return self.renderer(artifact)
        return render(artifact.value, artifact.kind)

    @param.depends("renderer", watch=True)
    def _rerender(self):
        self._rendered = None
        self._update_body()

    @param.depends("actions", "artifact_actions", watch=True)
    def refresh(self):
        """Evaluate again which actions apply to the artifact and which are done."""
        objects = []
        for spec in self.artifact_actions:
            if self.actions is None or spec.action not in self.actions:
                continue
            if spec.done is not None and (label := spec.done(self.artifact)):
                objects.append(pmui.Chip(
                    label=label, icon="check", size="small", color="success", variant="outlined",
                    margin=0, css_classes=["artifact-cell-done"],
                ))
                continue
            arguments = spec.arguments(self.artifact)
            if arguments is None:
                continue
            button = pmui.Button(
                label=spec.label, icon=spec.icon, variant="outlined", size="small", margin=0,
                sx={"textTransform": "none"}, css_classes=["artifact-cell-action"],
            )
            button.on_click(lambda _, s=spec, a=arguments, b=button: pn.state.execute(partial(self._invoke, s, a, b)))
            objects.append(button)
        self._actions.objects = objects

    async def _invoke(self, spec: ArtifactAction, arguments: dict[str, Any], button: pmui.Button):
        button.loading = True
        self._error.visible = False
        try:
            result = self.actions.invoke(spec.action, arguments)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.info("The %r action failed on %s: %s", spec.action, self.artifact.artifact_id, exc)
            self._error.param.update(object=f"{spec.label} failed: {exc}", visible=True)
        finally:
            button.loading = False
        self.refresh()


__all__ = ["ArtifactAction", "ArtifactCell"]
