"""
Defines SchemaForm, which generates a form of Material UI widgets from a
JSON schema and keeps a dict ``value`` in sync with them.

The supported subset covers what pydantic emits for models (``$defs`` and
``$ref``, ``anyOf`` with ``null`` for optional fields, ``allOf`` wrappers)
and the restricted schemas used by MCP elicitation (flat objects of
primitives with ``enum``, ``oneOf``/``const`` titled options and formats).
Shapes outside that subset fall back to a JSON editor for that field.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

import param
from panel.viewable import Viewer
from panel_material_ui.layout import Card, Column
from panel_material_ui.widgets import (
    Checkbox,
    DatePicker,
    DatetimePicker,
    FloatInput,
    IntInput,
    MultiChoice,
    PasswordInput,
    Select,
    Switch,
    TextAreaInput,
    TextInput,
)

TEXTAREA_FORMATS = {"textarea", "multiline", "markdown"}

TEXTAREA_NAMES = {"body", "content", "code", "query", "sql", "prompt", "script", "text", "message"}

TEXTAREA_MIN_LENGTH = 120

_MISSING = object()

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_NONE_LABEL = "(none)"


def resolve_schema(schema: dict[str, Any], root: dict[str, Any], depth: int = 0) -> tuple[dict[str, Any], bool]:
    """
    Normalize a schema node, returning ``(schema, nullable)``.

    Resolves local ``$ref`` pointers, merges ``allOf`` and unwraps
    ``anyOf``/``oneOf`` unions with ``null`` (pydantic's ``Optional``).
    Sibling keys such as ``title``, ``description`` and ``default``
    override those of the referenced schema.
    """
    if depth > 16:
        return {"x-unsupported": "recursive"}, False
    schema = dict(schema or {})
    nullable = False
    if "$ref" in schema:
        target = _lookup_ref(schema.pop("$ref"), root)
        target, nullable = resolve_schema(target, root, depth + 1)
        schema = {**target, **schema}
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for sub in schema.pop("allOf"):
            sub, sub_nullable = resolve_schema(sub, root, depth + 1)
            nullable = nullable or sub_nullable
            props = {**merged.get("properties", {}), **sub.get("properties", {})}
            required = [*merged.get("required", []), *sub.get("required", [])]
            merged = {**merged, **sub}
            if props:
                merged["properties"] = props
            if required:
                merged["required"] = required
        schema = {**merged, **schema}
    for key in ("anyOf", "oneOf"):
        if key not in schema:
            continue
        options = schema[key]
        rest = [o for o in options if o.get("type") != "null"]
        nullable = nullable or len(rest) < len(options)
        if rest and all("const" in o for o in rest):
            schema.pop(key)
            schema["enum"] = [o["const"] for o in rest]
            schema["x-enum-titles"] = [o.get("title", str(o["const"])) for o in rest]
            schema.setdefault("type", _infer_type(schema["enum"][0]))
        elif len(rest) == 1:
            schema.pop(key)
            sub, sub_nullable = resolve_schema(rest[0], root, depth + 1)
            nullable = nullable or sub_nullable
            schema = {**sub, **schema}
        elif nullable:
            schema[key] = rest
    types = schema.get("type")
    if isinstance(types, list):
        rest = [t for t in types if t != "null"]
        nullable = nullable or len(rest) < len(types)
        schema["type"] = rest[0] if len(rest) == 1 else rest
    if "default" in schema and schema["default"] is None:
        nullable = True
    return schema, nullable


def _lookup_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#"):
        return {"x-unsupported": f"external $ref {ref}"}
    node: Any = root
    for part in ref.lstrip("#/").split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return {"x-unsupported": f"unresolved $ref {ref}"}
        node = node[part]
    return node


def _infer_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def infer_schema(value: dict[str, Any]) -> dict[str, Any]:
    """Infer a permissive object schema from an example ``value``."""
    def infer(v):
        kind = _infer_type(v)
        if kind == "object":
            return {"type": "object", "properties": {k: infer(i) for k, i in v.items()}}
        if kind == "array":
            item_types = {_infer_type(i) for i in v}
            if len(item_types) == 1 and next(iter(item_types)) in ("string", "integer", "number", "boolean"):
                return {"type": "array", "items": {"type": item_types.pop()}}
            return {"type": "array"}
        if kind == "string" and ("\n" in v or len(v) > TEXTAREA_MIN_LENGTH):
            return {"type": "string", "format": "textarea"}
        return {"type": kind} if kind else {}
    return infer(dict(value or {}))


def _label(name: str, schema: dict[str, Any]) -> str:
    title = schema.get("title")
    if title:
        return title
    return name.replace("_", " ").strip().capitalize() if name else ""


def _parse_datetime(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _parse_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and value:
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


class _Field:
    """A single property of the schema, bound to one widget or a nested group."""

    def __init__(self, form: SchemaForm, name: str, path: tuple[str, ...], schema: dict[str, Any], required: bool):
        self.form = form
        self.name = name
        self.path = path
        self.schema, self.nullable = resolve_schema(schema, form.schema or {})
        self.required = required
        self.children: list[_Field] = []
        self.json_error: str | None = None
        self._last_json: Any = None
        self.explicit = False
        self.kind = self._kind()
        self.help = self._help()
        self.widget = self._make_widget()

    @property
    def label(self) -> str:
        label = _label(self.name, self.schema)
        return f"{label} *" if self.required else label

    def _kind(self) -> str:
        s = self.schema
        stype = s.get("type")
        fmt = s.get("format", "")
        if "x-unsupported" in s or isinstance(stype, list) or "anyOf" in s or "oneOf" in s:
            return "json"
        if "enum" in s and stype != "array":
            return "enum"
        if "const" in s:
            return "json"
        if stype == "string":
            if fmt == "date":
                return "date"
            if fmt == "date-time":
                return "datetime"
            if fmt == "password" or s.get("writeOnly"):
                return "password"
            if (
                fmt in TEXTAREA_FORMATS or s.get("contentMediaType") or self.name.lower() in TEXTAREA_NAMES
                or (s.get("maxLength") or 0) > TEXTAREA_MIN_LENGTH
            ):
                return "textarea"
            return "string"
        if stype in ("integer", "number", "boolean"):
            return stype
        if stype == "array":
            items, _ = resolve_schema(s.get("items", {}), self.form.schema or {})
            if "enum" in items:
                self.items = items
                return "multienum"
            return "array"
        if stype == "object" or (stype is None and "properties" in s):
            return "object" if s.get("properties") else "json"
        return "json"

    def _help(self) -> str:
        s = self.schema
        parts = [s["description"]] if s.get("description") else []
        lo = s.get("minimum", s.get("exclusiveMinimum"))
        hi = s.get("maximum", s.get("exclusiveMaximum"))
        if self.kind in ("integer", "number") and (lo is not None or hi is not None):
            if lo is not None and hi is not None:
                parts.append(f"Between {lo} and {hi}.")
            elif lo is not None:
                parts.append(f"At least {lo}.")
            else:
                parts.append(f"At most {hi}.")
        if self.kind == "json":
            parts.append("JSON value.")
        return " ".join(parts)

    def _options(self, schema: dict[str, Any]) -> dict[str, Any]:
        titles = schema.get("x-enum-titles") or schema.get("enumNames") or [str(v) for v in schema["enum"]]
        return dict(zip(titles, schema["enum"], strict=False))

    def _make_widget(self):
        kind = self.kind
        common = dict(label=self.label, disabled=self.form.disabled, sizing_mode="stretch_width")
        s = self.schema
        if kind == "object":
            required = set(s.get("required", []))
            self.children = [
                _Field(self.form, name, (*self.path, name), sub, name in required)
                for name, sub in s.get("properties", {}).items()
            ]
            header = self.help
            card = Card(
                *([] if not header else [_help_text(header)]),
                *(c.widget for c in self.children),
                title=self.label, collapsible=True, outlined=True,
                sizing_mode="stretch_width", margin=(5, 0), title_variant="subtitle1",
            )
            return card
        if kind == "enum":
            options = self._options(s)
            if self.nullable or not self.required:
                options = {_NONE_LABEL: None, **options}
            widget = Select(options=options, **common)
        elif kind == "multienum":
            widget = MultiChoice(options=self._options(self.items), **common)
        elif kind == "string":
            widget = TextInput(**({"max_length": s["maxLength"]} if "maxLength" in s else {}), **common)
        elif kind == "password":
            widget = PasswordInput(**common)
        elif kind == "textarea":
            widget = TextAreaInput(auto_grow=True, rows=3, max_rows=12, **common)
        elif kind in ("integer", "number"):
            cls = IntInput if kind == "integer" else FloatInput
            bounds = {}
            if "minimum" in s:
                bounds["start"] = s["minimum"]
            if "maximum" in s:
                bounds["end"] = s["maximum"]
            widget = cls(value=None, **bounds, **common)
        elif kind == "boolean":
            cls = Checkbox if self.form.boolean_widget == "checkbox" else Switch
            widget = cls(label=self.label, disabled=self.form.disabled, description=self.help)
        elif kind == "date":
            widget = DatePicker(value=None, clearable=True, **common)
        elif kind == "datetime":
            widget = DatetimePicker(value=None, clearable=True, **common)
        elif kind == "array":
            widget = TextInput(placeholder="[]", **common)
        else:
            widget = TextAreaInput(auto_grow=True, rows=2, max_rows=12, placeholder="null", **common)
        if "helper_text" in widget.param:
            widget.helper_text = self.help
        widget.param.watch(lambda event: self.form._on_widget_change(), "value")
        return widget

    def leaves(self):
        if self.kind == "object":
            for child in self.children:
                yield from child.leaves()
        else:
            yield self

    def get(self) -> Any:
        """Return the JSON value of the field, or ``_MISSING`` if it is empty."""
        kind = self.kind
        if kind == "object":
            value = {}
            for child in self.children:
                v = child.get()
                # Unset optional fields stay absent rather than becoming explicit nulls.
                if v is _MISSING or (v is None and not child.explicit):
                    continue
                value[child.name] = v
            return value if value or self.required else _MISSING
        w = self.widget.value
        if kind in ("string", "password", "textarea"):
            return w if w else (None if self.nullable else _MISSING)
        if kind == "boolean":
            return bool(w)
        if kind == "enum":
            return w if w is not None or self.nullable else _MISSING
        if kind in ("integer", "number"):
            if w is None:
                return None if self.nullable else _MISSING
            return int(w) if kind == "integer" else float(w)
        if kind == "date":
            return w.isoformat() if w else (None if self.nullable else _MISSING)
        if kind == "datetime":
            return w.isoformat() if w else (None if self.nullable else _MISSING)
        if kind == "multienum":
            return list(w or [])
        text = (w or "").strip()
        if not text:
            self.json_error = None
            return None if self.nullable else _MISSING
        try:
            value = json.loads(text)
        except ValueError as e:
            self.json_error = f"Invalid JSON: {e.msg}"
            return self._last_json
        self.json_error = None
        self._last_json = value
        return value

    def set(self, value: Any):
        kind = self.kind
        self.explicit = value is not _MISSING
        if value is _MISSING:
            value = self.schema.get("default", _MISSING)
        if kind == "object":
            value = value if isinstance(value, dict) else {}
            for child in self.children:
                child.set(value.get(child.name, _MISSING))
            return
        if value is _MISSING:
            value = None
        if kind in ("string", "password", "textarea"):
            new = "" if value is None else str(value)
        elif kind == "boolean":
            new = bool(value)
        elif kind in ("integer", "number"):
            new = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        elif kind == "date":
            new = _parse_date(value)
        elif kind == "datetime":
            new = _parse_datetime(value)
        elif kind == "enum":
            options = self.widget.options
            if value not in options.values():
                # Keep out-of-schema values visible so they are not silently dropped.
                options = {**options, str(value): value}
                self.widget.options = options
            new = value
        elif kind == "multienum":
            values = list(value or [])
            missing = [v for v in values if v not in self.widget.options.values()]
            if missing:
                self.widget.options = {**self.widget.options, **{str(v): v for v in missing}}
            new = values
        else:
            self._last_json = value
            self.json_error = None
            if value is None:
                new = ""
            else:
                new = json.dumps(value) if kind == "array" else json.dumps(value, indent=2, default=str)
        self.widget.value = new

    def errors(self, value: Any) -> list[str]:
        s = self.schema
        if self.kind == "object":
            if value is _MISSING:
                value = {}
            errors = []
            for child in self.children:
                errors += child.errors(value.get(child.name, _MISSING))
            return errors
        if self.json_error:
            return [self.json_error]
        if value is _MISSING or (value is None and not self.nullable):
            return ["Required."] if self.required else []
        if value is None:
            return []
        errors = []
        if self.kind == "boolean" and self.required and s.get("const") is True and value is not True:
            errors.append("Must be checked.")
        if "enum" in s and self.kind == "enum" and value not in s["enum"]:
            errors.append(f"Must be one of {', '.join(map(str, s['enum']))}.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in s and value < s["minimum"]:
                errors.append(f"Must be at least {s['minimum']}.")
            if "maximum" in s and value > s["maximum"]:
                errors.append(f"Must be at most {s['maximum']}.")
            if "exclusiveMinimum" in s and value <= s["exclusiveMinimum"]:
                errors.append(f"Must be greater than {s['exclusiveMinimum']}.")
            if "exclusiveMaximum" in s and value >= s["exclusiveMaximum"]:
                errors.append(f"Must be less than {s['exclusiveMaximum']}.")
            if "multipleOf" in s and s["multipleOf"] and (value / s["multipleOf"]) % 1:
                errors.append(f"Must be a multiple of {s['multipleOf']}.")
        if isinstance(value, str):
            if "minLength" in s and len(value) < s["minLength"]:
                errors.append(f"Must be at least {s['minLength']} characters.")
            if "maxLength" in s and len(value) > s["maxLength"]:
                errors.append(f"Must be at most {s['maxLength']} characters.")
            if "pattern" in s and not re.search(s["pattern"], value):
                errors.append(f"Must match pattern {s['pattern']}.")
            if s.get("format") == "email" and not _EMAIL.match(value):
                errors.append("Must be a valid email address.")
        if isinstance(value, list):
            if "minItems" in s and len(value) < s["minItems"]:
                errors.append(f"Must have at least {s['minItems']} items.")
            if "maxItems" in s and len(value) > s["maxItems"]:
                errors.append(f"Must have at most {s['maxItems']} items.")
            if self.kind == "multienum":
                invalid = [v for v in value if v not in self.items["enum"]]
                if invalid:
                    errors.append(f"Invalid options: {', '.join(map(str, invalid))}.")
        return errors

    def show_errors(self, errors: list[str]):
        widget = self.widget
        if "error_state" in widget.param:
            widget.param.update(error_state=bool(errors), helper_text=" ".join(errors) if errors else self.help)


def _help_text(text: str):
    from panel_material_ui.pane import Typography
    return Typography(text, variant="body2", color="text.secondary", margin=(0, 5, 5, 5))


class SchemaForm(Viewer):
    """
    The `SchemaForm` generates a form of Material UI widgets from a JSON
    schema and keeps `value` in sync with the widgets in both directions.

    Strings map to text inputs (or text areas for long text), enums to
    selects, numbers to numeric inputs honoring `minimum`/`maximum`,
    booleans to switches, `date`/`date-time` formats to pickers, enum
    arrays to multi-choice inputs and nested objects to grouped cards.
    `$ref`/`$defs`, `allOf` and nullable `anyOf` unions are resolved;
    any other shape falls back to a JSON editor for that field.

    Reference: https://panel-material-ui.holoviz.org/reference/widgets/SchemaForm.html

    :Example:

    >>> SchemaForm(schema={
    ...     "type": "object",
    ...     "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    ...     "required": ["name"],
    ... }, value={"name": "Ada"})
    """

    boolean_widget = param.Selector(default="switch", objects=["switch", "checkbox"], doc="""
        Widget used for boolean fields.""")

    disabled = param.Boolean(default=False, doc="Whether the form widgets are disabled.")

    errors = param.List(default=[], constant=True, doc="""
        Errors found by the most recent call to validate().""")

    live_validation = param.Boolean(default=False, doc="""
        Whether to validate (and display errors) on every change rather
        than only when validate() is called.""")

    schema = param.Dict(default={"type": "object", "properties": {}}, doc="""
        The JSON schema describing the form, usually an object schema.""")

    value = param.Dict(default={}, doc="""
        The current form value as JSON-compatible data.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._syncing = False
        self._layout = Column(sizing_mode="stretch_width", margin=0)
        self._root: _Field | None = None
        self._build()

    def __panel__(self):
        return self._layout

    @param.depends("schema", "boolean_widget", watch=True)
    def _build(self):
        schema = self.schema or {}
        root_schema, _ = resolve_schema(schema, schema)
        if root_schema.get("type", "object") != "object" or "properties" not in root_schema:
            # A free-form object becomes a single JSON editor.
            root_schema = {"type": "object", "properties": {"value": {"title": root_schema.get("title", "Value")}}}
            self._wrapped = True
        else:
            self._wrapped = False
        self._syncing = True
        try:
            self._root = _Field(self, "", (), root_schema, True)
        finally:
            self._syncing = False
        self._layout[:] = [c.widget for c in self._root.children]
        self._push_value()

    @param.depends("disabled", watch=True)
    def _update_disabled(self):
        for field in self._fields():
            field.widget.disabled = self.disabled

    @param.depends("value", watch=True)
    def _push_value(self):
        if self._syncing or self._root is None:
            return
        self._syncing = True
        try:
            self._root.set({"value": self.value} if self._wrapped else self.value)
            # Reflect defaults and coercions so value always matches the widgets.
            collected = self._collect()
            if collected != self.value:
                self.value = collected
        finally:
            self._syncing = False

    def _on_widget_change(self):
        if self._syncing or self._root is None:
            return
        value = self._collect()
        self._syncing = True
        try:
            self.value = value
        finally:
            self._syncing = False
        if self.live_validation:
            self.validate()

    def _collect(self) -> dict[str, Any]:
        value = self._root.get()
        value = {} if value is _MISSING else value
        if self._wrapped:
            inner = value.get("value")
            return inner if isinstance(inner, dict) else {}
        return value

    def _fields(self):
        if self._root is not None:
            yield from self._root.leaves()

    def field(self, path: str | tuple[str, ...]):
        """Return the widget for the field at `path`, a dotted string or tuple of keys."""
        parts = tuple(path.split(".")) if isinstance(path, str) else tuple(path)
        if self._wrapped:
            # The single JSON editor of a free-form schema is addressable as "" or "value".
            parts = ("value",)
        node = self._root
        for part in parts:
            node = next((c for c in node.children if c.name == part), None)
            if node is None:
                raise KeyError(path)
        return node.widget

    def validate(self) -> list[str]:
        """
        Validate the current value against the schema, display errors
        on the widgets and return them as ``"<path>: <message>"`` strings.
        """
        if self._root is None:
            return []
        value = self._root.get()
        errors = []

        def walk(field: _Field, v: Any):
            if field.kind == "object":
                if v is _MISSING and not field.required:
                    return
                v = v if isinstance(v, dict) else {}
                for child in field.children:
                    walk(child, v.get(child.name, _MISSING))
                return
            field_errors = field.errors(v)
            field.show_errors(field_errors)
            path = ".".join(field.path[1:] if self._wrapped else field.path) or "value"
            errors.extend(f"{path}: {e}" for e in field_errors)

        walk(self._root, value)
        with param.edit_constant(self):
            self.errors = errors
        return errors

    @property
    def valid(self) -> bool:
        return not self.validate()


__all__ = ["SchemaForm", "infer_schema", "resolve_schema"]
