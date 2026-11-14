from panel_agents.components._diff import DiffLine, DiffViewer, diff_lines, split_rows

OLD = "\n".join(f"line {i}" for i in range(20))
NEW = OLD.replace("line 3", "LINE 3").replace("line 15\n", "")


def test_diff_lines_identical():
    lines = diff_lines("a\nb", "a\nb")
    assert [line.op for line in lines] == ["equal", "equal"]


def test_diff_lines_none_and_empty():
    assert diff_lines(None, None) == []
    lines = diff_lines(None, "a\nb")
    assert [(line.op, line.new_lineno) for line in lines] == [("insert", 1), ("insert", 2)]
    lines = diff_lines("a", "")
    assert [(line.op, line.old_lineno) for line in lines] == [("delete", 1)]


def test_diff_lines_collapses_context():
    lines = diff_lines(OLD, NEW, context=3)
    skipped = [line for line in lines if isinstance(line, int)]
    assert skipped == [5]
    leading = lines[:3]
    assert [line.old_lineno for line in leading] == [1, 2, 3]
    changed = [line for line in lines if isinstance(line, DiffLine) and line.op != "equal"]
    assert [(line.op, line.text) for line in changed] == [
        ("delete", "line 3"), ("insert", "LINE 3"), ("delete", "line 15"),
    ]


def test_diff_lines_full_context():
    lines = diff_lines(OLD, NEW, context=None)
    assert not any(isinstance(line, int) for line in lines)
    assert len(lines) == 21


def test_diff_lines_collapses_leading_and_trailing():
    old = "\n".join(str(i) for i in range(10))
    new = old.replace("5", "five")
    lines = diff_lines(old, new, context=1)
    assert lines[0] == 4
    assert lines[-1] == 3


def test_split_rows_pairs_replacements():
    rows = split_rows(diff_lines("a\nb\nc", "a\nB\nC\nD\nc"))
    assert rows[0].old is rows[0].new
    assert (rows[1].old.text, rows[1].new.text) == ("b", "B")
    assert (rows[2].old, rows[2].new.text) == (None, "C")
    assert (rows[3].old, rows[3].new.text) == (None, "D")
    assert rows[4].old.text == "c"


def test_split_rows_keeps_skips():
    rows = split_rows(diff_lines(OLD, NEW, context=3))
    assert 5 in rows


def test_viewer_counts_and_html():
    viewer = DiffViewer(old=OLD, new=NEW, old_label="v1", new_label="<v2>")
    assert (viewer.added, viewer.removed) == (1, 2)
    html = viewer._html.object
    assert html.count('class="insert"') == 1
    assert html.count('class="delete"') == 2
    assert "5 unchanged lines" in html
    assert "&lt;v2&gt;" in html


def test_viewer_escapes_content():
    viewer = DiffViewer(old="<b>", new="<i>")
    assert "&lt;i&gt;" in viewer._html.object
    assert "<i>" not in viewer._html.object


def test_viewer_updates():
    viewer = DiffViewer(old="a", new="a")
    assert "No differences" in viewer._html.object
    viewer.new = "b"
    assert (viewer.added, viewer.removed) == (1, 1)
    viewer.mode = "split"
    assert 'td class="code insert"' in viewer._html.object
    viewer.context = None
    viewer.show_header = False
    assert "diff-header" not in viewer._html.object


def test_viewer_options():
    viewer = DiffViewer(old=OLD, new=NEW, wrap=True, max_height=200)
    assert 'class="diff wrap"' in viewer._html.object
    assert "max-height: 200px" in viewer._html.object


def test_viewer_renders(document, comm):
    viewer = DiffViewer(old="a", new="b")
    model = viewer.get_root(document, comm)
    assert model is not None
