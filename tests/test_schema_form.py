import datetime as dt
from typing import Literal, Optional

import pytest
from panel_material_ui.widgets import (
    DatePicker,
    DatetimePicker,
    FloatInput,
    IntInput,
    MultiChoice,
    Select,
    Switch,
    TextAreaInput,
    TextInput,
)
from pydantic import BaseModel, Field

from panel_agents.components._schema_form import SchemaForm, infer_schema, resolve_schema


class Address(BaseModel):
    city: str
    zip_code: Optional[str] = None


class Query(BaseModel):
    """Run a SQL query."""

    sql: str = Field(description="The SQL to execute.")
    limit: int = Field(100, ge=1, le=1000)
    ratio: float = 0.5
    mode: Literal["fast", "safe"] = "safe"
    tags: list[Literal["a", "b"]] = []
    since: Optional[dt.date] = None
    at: Optional[dt.datetime] = None
    dry_run: bool = False
    address: Optional[Address] = None
    columns: list[str] = []
    extra: int | str = 1


ELICITATION = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "title": "Your name", "minLength": 2},
        "email": {"type": "string", "format": "email"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "color": {"type": "string", "enum": ["r", "g"], "enumNames": ["Red", "Green"]},
        "size": {"type": "string", "oneOf": [{"const": "s", "title": "Small"}, {"const": "l", "title": "Large"}]},
        "subscribe": {"type": "boolean", "default": True},
        "birthday": {"type": "string", "format": "date"},
        "toppings": {"type": "array", "items": {"anyOf": [{"const": "ham", "title": "Ham"}, {"const": "egg", "title": "Egg"}]}},
    },
    "required": ["name", "email"],
}


def test_resolve_schema_ref_and_optional():
    schema = Query.model_json_schema()
    resolved, nullable = resolve_schema(schema["properties"]["address"], schema)
    assert nullable
    assert resolved["type"] == "object"
    assert set(resolved["properties"]) == {"city", "zip_code"}


def test_resolve_schema_type_list():
    resolved, nullable = resolve_schema({"type": ["integer", "null"]}, {})
    assert resolved["type"] == "integer"
    assert nullable


def test_pydantic_schema_widgets():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "SELECT 1"})
    assert isinstance(form.field("sql"), TextAreaInput)
    assert form.field("sql").label == "Sql *"
    assert form.field("sql").helper_text == "The SQL to execute."
    limit = form.field("limit")
    assert isinstance(limit, IntInput)
    assert (limit.start, limit.end) == (1, 1000)
    assert isinstance(form.field("ratio"), FloatInput)
    assert isinstance(form.field("mode"), Select)
    assert isinstance(form.field("tags"), MultiChoice)
    assert isinstance(form.field("since"), DatePicker)
    assert isinstance(form.field("at"), DatetimePicker)
    assert isinstance(form.field("dry_run"), Switch)
    assert isinstance(form.field("address.city"), TextInput)
    assert form.field("address.city").label == "City *"
    assert isinstance(form.field("columns"), TextInput)
    assert isinstance(form.field("extra"), TextAreaInput)


def test_defaults_fill_value_and_unset_optionals_are_omitted():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "SELECT 1"})
    assert form.value == {
        "sql": "SELECT 1", "limit": 100, "ratio": 0.5, "mode": "safe", "tags": [],
        "dry_run": False, "columns": [], "extra": 1,
    }
    assert form.field("limit").value == 100


def test_widget_to_value():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "SELECT 1"})
    form.field("limit").value = 5
    form.field("since").value = dt.date(2024, 1, 2)
    form.field("at").value = dt.datetime(2024, 1, 2, 3, 4)
    form.field("address.city").value = "Berlin"
    form.field("columns").value = '["a", "b"]'
    form.field("tags").value = ["a"]
    assert form.value["limit"] == 5
    assert form.value["since"] == "2024-01-02"
    assert form.value["at"] == "2024-01-02T03:04:00"
    assert form.value["address"] == {"city": "Berlin"}
    assert form.value["columns"] == ["a", "b"]
    assert form.value["tags"] == ["a"]


def test_value_to_widgets():
    form = SchemaForm(schema=Query.model_json_schema())
    form.value = {"sql": "SELECT 2", "limit": 7, "mode": "fast", "since": "2023-05-06", "address": {"city": "Oslo"}}
    assert form.field("sql").value == "SELECT 2"
    assert form.field("limit").value == 7
    assert form.field("mode").value == "fast"
    assert form.field("since").value == dt.date(2023, 5, 6)
    assert form.field("address.city").value == "Oslo"
    assert form.value["address"] == {"city": "Oslo"}
    assert form.value["ratio"] == 0.5


def test_explicit_null_is_preserved():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "x", "since": None})
    assert "since" in form.value and form.value["since"] is None


def test_validate_required_and_bounds():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "", "limit": 5000})
    errors = form.validate()
    assert "sql: Required." in errors
    assert "limit: Must be at most 1000." in errors
    assert form.field("limit").error_state
    assert form.errors == errors
    form.field("limit").value = 10
    form.field("sql").value = "SELECT 1"
    assert form.validate() == []
    assert not form.field("limit").error_state
    assert form.field("sql").helper_text == "The SQL to execute."


def test_validate_skips_unset_optional_object():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "x"})
    assert form.validate() == []
    form.field("address.zip_code").value = "123"
    assert form.validate() == ["address.city: Required."]


def test_invalid_json_field():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "x", "columns": ["a"]})
    form.field("columns").value = "[1,"
    assert form.value["columns"] == ["a"]
    assert form.validate()[0].startswith("columns: Invalid JSON")


def test_elicitation_schema():
    form = SchemaForm(schema=ELICITATION)
    assert form.field("name").label == "Your name *"
    assert isinstance(form.field("email"), TextInput)
    color = form.field("color")
    assert isinstance(color, Select)
    assert color.options == {"(none)": None, "Red": "r", "Green": "g"}
    assert form.field("size").options == {"(none)": None, "Small": "s", "Large": "l"}
    toppings = form.field("toppings")
    assert isinstance(toppings, MultiChoice)
    assert toppings.options == {"Ham": "ham", "Egg": "egg"}
    assert form.field("subscribe").value is True
    assert form.value == {"subscribe": True, "toppings": []}

    form.field("name").value = "A"
    form.field("email").value = "nope"
    form.field("age").value = 200
    assert form.validate() == [
        "name: Must be at least 2 characters.",
        "email: Must be a valid email address.",
        "age: Must be at most 150.",
    ]
    form.param.update(value={"name": "Ada", "email": "ada@example.com", "color": "g", "size": "l", "toppings": ["egg"]})
    assert form.validate() == []
    assert form.field("color").value == "g"
    assert form.field("toppings").value == ["egg"]


def test_out_of_schema_enum_value_is_kept_and_flagged():
    form = SchemaForm(schema={"type": "object", "properties": {"mode": {"enum": ["a", "b"]}}}, value={"mode": "c"})
    assert form.value == {"mode": "c"}
    assert form.validate() == ["mode: Must be one of a, b."]


def test_free_form_schema_falls_back_to_json_editor():
    form = SchemaForm(schema={"type": "object"}, value={"a": [1, {"b": None}]})
    editor = form.field("value")
    assert isinstance(editor, TextAreaInput)
    editor.value = '{"x": true}'
    assert form.value == {"x": True}


def test_infer_schema():
    schema = infer_schema({"to": "a@b.c", "count": 2, "body": "line\nline", "tags": ["x"], "meta": {"k": 1.5}})
    form = SchemaForm(schema=schema, value={"to": "a@b.c", "count": 2, "body": "line\nline", "tags": ["x"], "meta": {"k": 1.5}})
    assert isinstance(form.field("count"), IntInput)
    assert isinstance(form.field("body"), TextAreaInput)
    assert isinstance(form.field("meta.k"), FloatInput)
    assert form.value["tags"] == ["x"]


def test_schema_change_rebuilds():
    form = SchemaForm(schema={"type": "object", "properties": {"a": {"type": "string"}}}, value={"a": "x"})
    form.schema = {"type": "object", "properties": {"b": {"type": "integer", "default": 3}}}
    with pytest.raises(KeyError):
        form.field("a")
    assert form.value == {"b": 3}


def test_disabled_propagates():
    form = SchemaForm(schema=Query.model_json_schema(), value={"sql": "x"})
    form.disabled = True
    assert form.field("sql").disabled
    assert form.field("address.city").disabled


def test_live_validation():
    form = SchemaForm(schema=ELICITATION, live_validation=True)
    form.field("age").value = -1
    assert "age: Must be at least 0." in form.errors


def test_schema_form_is_public():
    import panel_agents
    from panel_agents.components import SchemaForm as Public

    assert panel_agents.SchemaForm is Public is SchemaForm
    form = Public(schema={"type": "object", "properties": {"n": {"type": "integer"}}}, value={"n": 2})
    assert form.value == {"n": 2} and form.valid


def test_field_returns_the_public_widget():
    form = SchemaForm(schema={
        "type": "object",
        "properties": {"n": {"type": "integer"}, "inner": {"type": "object", "properties": {"s": {"type": "string"}}}},
    })
    assert isinstance(form.field("n"), IntInput)
    assert isinstance(form.field("inner.s"), TextInput)
    assert type(form.field("inner")).__module__.startswith("panel_material_ui")
