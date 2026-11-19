"""
A compact picker of the model an agent runs its next request with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import param
from panel.custom import JSComponent
from panel.viewable import Viewer


@dataclass(frozen=True)
class ModelChoice:
    """
    A model a person may pick for the agent's next request.

    ``id`` names the model the way the runner's framework does, e.g. a
    pydantic-ai model string such as ``"openai:gpt-5"``. ``model``
    optionally holds a ready-made model object the runner uses instead of
    the id, e.g. one with a custom provider; the id then only identifies
    the choice. ``description`` is a short hint, e.g. about cost or speed.
    """

    id: str
    label: str | None = None
    description: str | None = None
    model: Any = None

    @property
    def name(self) -> str:
        return self.label or self.id

    @property
    def target(self) -> Any:
        """What the runner is given: the model object if there is one, else the id."""
        return self.model if self.model is not None else self.id

    @classmethod
    def coerce(cls, spec: ModelChoice | str | dict[str, Any]) -> ModelChoice:
        """Make a ModelChoice from a model id, a dict of its fields or a ModelChoice."""
        if isinstance(spec, ModelChoice):
            return spec
        if isinstance(spec, str):
            return cls(spec)
        if isinstance(spec, dict):
            return cls(**spec)
        raise TypeError(f"A model choice is a model id, a dict or a ModelChoice, not {type(spec).__name__}.")


_MENU_ESM = """
const CARET = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>'

export function render({ model, el }) {
  const button = document.createElement("button")
  button.type = "button"
  button.className = "pa-model-button"
  button.setAttribute("aria-haspopup", "listbox")
  button.setAttribute("aria-expanded", "false")
  const current = document.createElement("span")
  current.className = "pa-model-current"
  button.append(current)
  button.insertAdjacentHTML("beforeend", CARET)
  const list = document.createElement("div")
  list.className = "pa-model-list"
  list.setAttribute("role", "listbox")
  list.hidden = true
  el.append(button, list)

  const items = () => [...list.querySelectorAll('[role="option"]')]

  const update = () => {
    // A host may not apply the component's visibility, e.g. in a chat input's footer, so it hides itself too.
    el.style.display = model.options.length > 1 ? "" : "none"
    const option = model.options.find(o => o.id === model.value)
    current.textContent = option ? option.label : ""
    button.setAttribute("aria-label", `${model.label}: ${option ? option.label : "none"}`)
    list.setAttribute("aria-label", model.label)
    list.replaceChildren(...model.options.map(o => {
      const item = document.createElement("div")
      item.className = "pa-model-option"
      item.setAttribute("role", "option")
      item.dataset.id = o.id
      item.tabIndex = -1
      item.setAttribute("aria-selected", String(o.id === model.value))
      const primary = document.createElement("span")
      primary.className = "pa-model-primary"
      primary.textContent = o.label
      item.append(primary)
      const secondary = o.reason || o.description
      if (secondary) {
        const line = document.createElement("span")
        line.className = "pa-model-secondary"
        line.textContent = secondary
        item.append(line)
      }
      if (o.reason) { item.setAttribute("aria-disabled", "true") }
      return item
    }))
  }

  const close = (refocus) => {
    if (list.hidden) { return }
    list.hidden = true
    button.setAttribute("aria-expanded", "false")
    if (refocus) { button.focus() }
  }
  const open = () => {
    const rect = button.getBoundingClientRect()
    list.hidden = false
    const width = list.getBoundingClientRect().width
    list.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - width - 8))}px`
    list.style.bottom = `${window.innerHeight - rect.top + 4}px`
    button.setAttribute("aria-expanded", "true")
    const target = items().find(i => i.getAttribute("aria-selected") === "true") || items().find(i => i.getAttribute("aria-disabled") !== "true")
    if (target) { target.focus() }
  }
  const pick = (item) => {
    if (!item || item.getAttribute("aria-disabled") === "true") { return }
    model.value = item.dataset.id
    close(true)
  }

  button.addEventListener("click", () => list.hidden ? open() : close(true))
  // Handled here, since a host such as a chat input may cancel Enter before it becomes a click.
  button.addEventListener("keydown", (e) => {
    if (["ArrowUp", "ArrowDown", "Enter", " "].includes(e.key)) {
      e.preventDefault()
      e.stopPropagation()
      if (list.hidden) { open() } else if (e.key === "Enter" || e.key === " ") { close(true) }
    }
  })
  list.addEventListener("click", (e) => pick(e.target.closest('[role="option"]')))
  list.addEventListener("keydown", (e) => {
    e.stopPropagation()
    if (e.key === "Home" || e.key === "End") {
      e.preventDefault()
      const usable = items().filter(i => i.getAttribute("aria-disabled") !== "true")
      const target = e.key === "Home" ? usable[0] : usable[usable.length - 1]
      if (target) { target.focus() }
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault()
      const all = items()
      const step = e.key === "ArrowDown" ? 1 : all.length - 1
      let at = all.indexOf(el.getRootNode().activeElement)
      for (let n = 0; n < all.length; n++) {
        at = (at + step + all.length) % all.length
        if (all[at].getAttribute("aria-disabled") !== "true") { all[at].focus(); break }
      }
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault()
      pick(el.getRootNode().activeElement)
    } else if (e.key === "Escape" || e.key === "Tab") {
      close(e.key === "Escape")
    }
  })
  const outside = (e) => { if (!e.composedPath().includes(el)) { close(false) } }
  document.addEventListener("pointerdown", outside)
  model.on("remove", () => document.removeEventListener("pointerdown", outside))

  update()
  model.on(["options", "value", "label"], update)

  try {
    const stored = model.storage_key && window.localStorage.getItem(model.storage_key)
    if (stored) { model.stored = stored }
  } catch (e) {}
  model.on("value", () => {
    try {
      if (model.storage_key && model.value) { window.localStorage.setItem(model.storage_key, model.value) }
    } catch (e) {}
  })
}
"""

_MENU_CSS = """
:host { display: block; }
.pa-model-button {
  display: inline-flex; align-items: center; gap: 2px; max-width: 220px; padding: 2px 2px 2px 0;
  border: 0; border-radius: var(--mui-shape-borderRadius, 4px); background: transparent; cursor: pointer;
  color: var(--mui-palette-text-secondary, rgba(0, 0, 0, 0.6));
  font: 400 0.8125rem/1.5 var(--mui-font-family, Roboto, sans-serif);
}
.pa-model-current { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pa-model-button:hover { color: var(--mui-palette-text-primary, rgba(0, 0, 0, 0.87)); }
.pa-model-button:focus-visible { outline: 2px solid currentColor; outline-offset: 1px; }
.pa-model-list {
  position: fixed; z-index: 1500; min-width: 200px; max-width: 320px; max-height: 50vh; overflow-y: auto;
  padding: 4px 0; box-sizing: border-box; border-radius: var(--mui-shape-borderRadius, 4px);
  background: var(--mui-palette-background-paper, #fff); color: var(--mui-palette-text-primary, rgba(0, 0, 0, 0.87));
  box-shadow: var(--mui-shadows-8, 0 5px 5px -3px rgba(0,0,0,.2), 0 8px 10px 1px rgba(0,0,0,.14), 0 3px 14px 2px rgba(0,0,0,.12));
  background-image: var(--mui-overlays-8, none);
  font: 400 0.875rem/1.43 var(--mui-font-family, Roboto, sans-serif);
}
.pa-model-list[hidden] { display: none; }
.pa-model-option { display: flex; flex-direction: column; padding: 6px 16px; cursor: pointer; outline: none; }
.pa-model-option:hover, .pa-model-option:focus { background: var(--mui-palette-action-hover, rgba(0, 0, 0, 0.04)); }
.pa-model-option[aria-selected="true"] { background: var(--mui-palette-action-selected, rgba(0, 0, 0, 0.08)); }
.pa-model-option[aria-disabled="true"] { cursor: default; }
.pa-model-option[aria-disabled="true"] .pa-model-primary { opacity: 0.5; }
.pa-model-secondary { font-size: 0.75rem; color: var(--mui-palette-text-secondary, rgba(0, 0, 0, 0.6)); }
"""


class ModelMenu(JSComponent):
    """
    The picker's button and list of models. A native button and listbox,
    so it has an accessible name and lists each model's description, or
    why it cannot be used, under its label. It also keeps the pick in the
    browser's localStorage under ``storage_key``.
    """

    options = param.List(default=[], doc="Dicts with an ``id``, a ``label``, an optional ``description`` and ``reason``.")

    value = param.String(default=None, allow_None=True)

    label = param.String(default="Model", doc="The accessible name of the button and the list.")

    storage_key = param.String(default="")

    stored = param.String(default=None, allow_None=True, doc="The id the browser had stored when it loaded.")

    _esm = _MENU_ESM

    _stylesheets = [_MENU_CSS]


class ModelPicker(Viewer):
    """
    A small menu of the models the agent may run with, for a chat's composer.

    The picked model is set as the runner's ``model``, which an
    ``AgentRunner`` reads when a request starts, so a change applies from
    the next request and never to one that is running. Models the runner's
    ``unavailable_reason`` finds unusable, e.g. because the provider's API key is
    missing, are listed disabled with the reason under their label, and a
    model's ``description`` is shown under its label too.
    The picker is hidden unless there is more than one model to choose from.
    """

    models = param.List(default=[], doc="""
        The models to choose from: model ids, dicts of ``ModelChoice``
        fields or ``ModelChoice`` objects.""")

    value = param.String(default=None, allow_None=True, doc="""
        The id of the picked model. Defaults to the first usable model.""")

    runner = param.Parameter(default=None, doc="""
        The runner whose ``model`` follows the picked model, e.g. an
        ``AgentRunner``. Its ``unavailable_reason(model)``, if it has one, tells
        why a model cannot be used.""")

    remember = param.String(default=None, allow_None=True, doc="""
        A localStorage key under which the browser remembers the picked
        model, so the next session starts with it if it is still offered
        and usable. None remembers nothing.""")

    unavailable = param.Dict(default={}, constant=True, doc="Why each unusable model cannot be used, by id.")

    label = param.String(default="Model", doc="The accessible name of the picker.")

    def __init__(self, **params):
        super().__init__(**params)
        self._menu = ModelMenu(label=self.label, storage_key=self.remember or "", margin=0)
        self._menu.param.watch(self._on_select, "value")
        self._menu.param.watch(self._on_stored, "stored")
        self._sync_models()

    def __panel__(self):
        return self._menu

    @property
    def choices(self) -> list[ModelChoice]:
        return self._choices

    def _check(self, choice: ModelChoice) -> str | None:
        check = getattr(self.runner, "unavailable_reason", None)
        if not callable(check):
            return None
        try:
            return check(choice.target)
        except Exception as e:
            return str(e) or type(e).__name__

    @param.depends("models", watch=True)
    def _sync_models(self):
        choices: dict[str, ModelChoice] = {}
        for spec in self.models:
            choice = ModelChoice.coerce(spec)
            choices.setdefault(choice.id, choice)
        self._choices = list(choices.values())
        reasons = {c.id: reason for c in self._choices if (reason := self._check(c))}
        with param.edit_constant(self):
            self.unavailable = reasons
        names = [c.name for c in self._choices]
        options = []
        for c in self._choices:
            # Models sharing a label are told apart by their ids.
            label = f"{c.name} ({c.id})" if names.count(c.name) > 1 and c.name != c.id else c.name
            option = {"id": c.id, "label": label}
            if c.description:
                option["description"] = c.description
            if c.id in reasons:
                option["reason"] = f"Unavailable: {reasons[c.id]}"
            options.append(option)
        usable = self._usable
        value = self.value if self.value in usable else (usable[0] if usable else None)
        self._menu.param.update(options=options, value=value, visible=len(self._choices) > 1)
        if value != self.value:
            self.value = value
        else:
            self._apply()

    @property
    def _usable(self) -> list[str]:
        return [c.id for c in self._choices if c.id not in self.unavailable]

    @param.depends("value", watch=True)
    def _apply(self):
        usable = self._usable
        if self.value not in usable and (self.value is not None or usable):
            # An id that is not offered or not usable falls back to the default.
            self.value = usable[0] if usable else None
            return
        self._menu.value = self.value
        choice = next((c for c in self._choices if c.id == self.value), None)
        if self.runner is not None and choice is not None:
            self.runner.model = choice.target

    def _on_select(self, event):
        if event.new in self._usable:
            self.value = event.new

    def _on_stored(self, event):
        if event.new in self._usable:
            self.value = event.new

    @param.depends("remember", "label", watch=True)
    def _sync_menu(self):
        self._menu.param.update(storage_key=self.remember or "", label=self.label)


__all__ = ["ModelChoice", "ModelPicker"]
