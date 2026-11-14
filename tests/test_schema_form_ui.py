import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents.components._schema_form import SchemaForm

pytestmark = pytest.mark.ui

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "title": "Name", "description": "Your full name."},
        "count": {"type": "integer", "minimum": 0, "maximum": 10},
        "color": {"type": "string", "enum": ["red", "green"]},
        "agree": {"type": "boolean", "title": "Agree"},
    },
    "required": ["name"],
}


def test_schema_form_widget_to_value(page):
    form = SchemaForm(schema=SCHEMA, value={"count": 1, "color": "red"})
    serve_component(page, form)

    expect(page.get_by_text("Your full name.")).to_be_visible()
    name = page.locator("input").first
    name.fill("Ada")
    name.press("Enter")
    wait_until(lambda: form.value.get("name") == "Ada", page)

    count = page.locator("input[type=text]").nth(1)
    count.fill("7")
    count.press("Tab")
    wait_until(lambda: form.value.get("count") == 7, page)


def test_schema_form_value_to_widget_and_errors(page):
    form = SchemaForm(schema=SCHEMA, value={"count": 1})
    serve_component(page, form)

    form.value = {"name": "Grace", "count": 20}
    expect(page.locator("input").first).to_have_value("Grace")
    assert form.validate() == ["count: Must be at most 10."]
    expect(page.get_by_text("Must be at most 10.")).to_be_visible()
