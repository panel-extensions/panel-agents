"""
Actions on the user interface that an agent can take on the user's behalf.

A host app registers the things a user can do in it, such as navigating,
selecting or filtering, as typed functions on an ``ActionRegistry``. The
registry derives a JSON schema from each signature, so the same actions
can be listed in a command palette, exposed to an agent as tools or
served over MCP without the descriptions drifting apart. Actions always
run in the Panel session that created the registry.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import param
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

_SECTION = re.compile(r"^(\w[\w ]*):\s*$")
_ARG = re.compile(r"^\*{0,2}(\w+)\s*(\([^)]*\))?\s*:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """
    Split a Google style docstring into its description and the
    descriptions of the arguments listed under ``Args:``.
    """
    description: list[str] = []
    args: dict[str, str] = {}
    section = None
    current = None
    args_indent = 0
    for line in inspect.cleandoc(doc or "").splitlines():
        if (header := _SECTION.match(line)) is not None:
            section, current = header.group(1).lower(), None
            continue
        if section is None:
            description.append(line)
        elif section in ("args", "arguments", "parameters") and line.strip():
            stripped = line.strip()
            match = _ARG.match(stripped)
            if match and (current is None or len(line) - len(line.lstrip()) <= args_indent):
                current, args_indent = match.group(1), len(line) - len(line.lstrip())
                args[current] = match.group(3).strip()
            elif current is not None:
                args[current] = f"{args[current]} {stripped}".strip()
    return "\n".join(description).strip(), args


@dataclass
class Action:
    """
    A named, typed function the user, a command palette or an agent can
    invoke. ``parameters`` is the JSON schema of its arguments, derived
    from the function's signature and docstring.
    """

    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any] = field(init=False, repr=False)
    _model: type[BaseModel] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _, arg_docs = parse_docstring(self.function.__doc__)
        signature = inspect.signature(self.function)
        hints = _type_hints(self.function)
        fields: dict[str, Any] = {}
        for pname, p in signature.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(f"Action {self.name!r} cannot take *{pname}; declare each argument.")
            annotation = hints.get(pname, Any)
            default = ... if p.default is p.empty else p.default
            fields[pname] = (annotation, _with_description(default, arg_docs.get(pname)))
        model_name = "".join(part.title() for part in self.name.split("_")) or "Action"
        self._model = create_model(model_name, __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True), **fields)
        schema = self._model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        self.parameters = schema

    @classmethod
    def from_function(cls, function: Callable[..., Any], name: str | None = None, description: str | None = None) -> Action:
        """An action named after ``function`` and described by its docstring unless given."""
        doc_description, _ = parse_docstring(function.__doc__)
        return cls(
            name=name or function.__name__,
            description=description if description is not None else doc_description,
            function=function,
        )

    def validate(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate ``args`` against the signature, returning the keyword arguments to call with."""
        instance = self._model.model_validate(args or {})
        return {name: getattr(instance, name) for name in type(instance).model_fields}

    def validate_json(self, data: str | bytes) -> dict[str, Any]:
        instance = self._model.model_validate_json(data or "{}")
        return {name: getattr(instance, name) for name in type(instance).model_fields}


def _type_hints(function) -> dict[str, Any]:
    try:
        return inspect.get_annotations(function, eval_str=True)
    except Exception:
        return {k: v for k, v in getattr(function, "__annotations__", {}).items() if not isinstance(v, str)}


def _with_description(default: Any, description: str | None):
    if not description or isinstance(default, FieldInfo):
        return default
    return Field(default, description=description)


def _server_loop(doc) -> asyncio.AbstractEventLoop | None:
    try:
        return doc.session_context.server_context.application_context.io_loop.asyncio_loop
    except AttributeError:
        return None


def _session_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class ActionRegistry(param.Parameterized):
    """
    The actions a host app offers in one user session.

    Create it inside the session, e.g. while building the page, so
    actions invoked from elsewhere (a tool running in another thread, an
    MCP request) still run in the user's session and update the page
    they are looking at. Calls from outside the session's event loop are
    handed to it through the document's next-tick callback, which also
    holds the document lock, so sync actions block that session while
    they run. Once the session is destroyed, invoking raises.

    >>> actions = ActionRegistry()
    >>> @actions.action
    ... def open_page(path: str) -> str:
    ...     '''Navigate to a page of the app.'''
    ...     return f"Opened {path}"
    """

    actions = param.Dict(default={}, constant=True, doc="""
        Mapping of action name to Action, in registration order.""")

    def __init__(self, **params):
        super().__init__(**params)
        import panel as pn

        self._doc = pn.state.curdoc
        self._loop = _session_loop()
        self._closed = False
        self._pending: set[concurrent.futures.Future] = set()
        if self._doc is not None and self._doc.session_context is not None:
            pn.state.on_session_destroyed(self._on_session_destroyed)

    def _on_session_destroyed(self, session_context):
        self._closed = True
        # Calls handed to the session that will now never run must not leave their callers waiting.
        for future in list(self._pending):
            if not future.done():
                future.set_exception(RuntimeError("The session these actions belong to has ended."))
        self._pending.clear()

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions.values())

    def __len__(self) -> int:
        return len(self.actions)

    def __contains__(self, name: str) -> bool:
        return name in self.actions

    def __getitem__(self, name: str) -> Action:
        return self.actions[name]

    def add(self, function: Callable[..., Any], *, name: str | None = None, description: str | None = None) -> Action:
        """Register ``function`` as an action, replacing any action of the same name."""
        action = Action.from_function(function, name=name, description=description)
        with param.edit_constant(self):
            self.actions = {**self.actions, action.name: action}
        return action

    def action(self, function: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
        """Decorator registering a function as an action, with or without arguments."""
        def register(f):
            self.add(f, name=name, description=description)
            return f
        return register if function is None else register(function)

    def remove(self, name: str):
        """Unregister an action."""
        with param.edit_constant(self):
            self.actions = {k: v for k, v in self.actions.items() if k != name}

    async def invoke(self, name: str, args: dict[str, Any] | None = None) -> Any:
        """
        Validate ``args`` and run the action in the registry's session,
        returning its result. Coroutine functions are awaited.
        """
        action = self.actions.get(name)
        if action is None:
            raise KeyError(f"Unknown action {name!r}; available: {', '.join(self.actions) or 'none'}.")
        return await self._invoke_validated(action, action.validate(args))

    async def _invoke_validated(self, action: Action, kwargs: dict[str, Any]) -> Any:
        if self._closed:
            raise RuntimeError(f"The session the {action.name!r} action belongs to has ended.")
        doc = self._doc
        # Resolved now rather than at construction, which may have happened on a worker thread.
        loop = _server_loop(doc) or self._loop
        if loop is None or not loop.is_running() or loop is _session_loop():
            return await self._call(action, kwargs)
        result: concurrent.futures.Future = concurrent.futures.Future()
        self._pending.add(result)
        result.add_done_callback(self._pending.discard)

        async def run():
            if result.done():
                return
            try:
                value = await self._call(action, kwargs)
            except BaseException as e:
                if not result.done():
                    result.set_exception(e)
            else:
                if not result.done():
                    result.set_result(value)

        if doc is not None and doc.session_context is not None:
            doc.add_next_tick_callback(run)
        else:
            asyncio.run_coroutine_threadsafe(run(), loop)
        return await asyncio.wrap_future(result)

    async def _call(self, action: Action, kwargs: dict[str, Any]) -> Any:
        from panel.io.state import set_curdoc

        context = set_curdoc(self._doc) if self._doc is not None else contextlib.nullcontext()
        with context:
            result = action.function(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        return result


__all__ = ["Action", "ActionRegistry"]
