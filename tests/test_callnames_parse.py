"""Tests for callnames.parse — per-app AX tree parsers.

Each parser is tested with synthetic AX node dicts.  Unknown tree shape
→ empty participants (never raises, never guesses).

All tests pass the tree as a list of root nodes (matching ax.sample() output).
"""

from __future__ import annotations

import unittest

from trnscrb.callnames import parse as _parse


def _node(**kw):
    """Convenience: AX node dict."""
    return {"role": "AXStaticText", "description": "", "title": "", "value": "", "children": [], **kw}


def _root(children):
    """Root node wrapping a list of children — returned as a single-element list."""
    return [
        {"role": "AXWindow", "description": "", "title": "", "value": "", "children": children}
    ]


# ── Zoom ──────────────────────────────────────────────────────────────────────

class TestZoom(unittest.TestCase):
    def test_video_tile_with_speaker(self):
        ann = _node(
            role="AXTabGroup",
            description="Ann Lee, Computer audio unmuted, Active speaker",
        )
        result = _parse.zoom(_root([ann]))
        self.assertIn("Ann Lee", result.participants)
        self.assertIn("Ann Lee", result.speaking)

    def test_video_tile_no_speaker(self):
        ann = _node(
            role="AXTabGroup",
            description="Ann Lee, Computer audio unmuted, Video off",
        )
        result = _parse.zoom(_root([ann]))
        self.assertIn("Ann Lee", result.participants)
        self.assertNotIn("Ann Lee", result.speaking)

    def test_participants_list(self):
        outline = _node(
            role="AXOutline",
            description="Participants list",
            children=[
                _node(role="AXRow", description="Ann Lee"),
                _node(role="AXRow", description="Bo Park (me)"),
            ],
        )
        result = _parse.zoom(_root([outline]))
        self.assertIn("Ann Lee", result.participants)
        self.assertIn("Bo Park", result.participants)
        self.assertEqual(result.me, "Bo Park")

    def test_talking_window(self):
        text = _node(role="AXStaticText", description="")
        text["value"] = "Talking: Ann Lee"
        result = _parse.zoom(_root([text]))
        self.assertIn("Ann Lee", result.speaking)

    def test_unknown_tree(self):
        result = _parse.zoom(_root([_node(role="AXButton", description="click me")]))
        self.assertEqual(result.participants, [])
        self.assertEqual(result.speaking, [])

    def test_mixed_linguistic(self):
        tile = _node(
            role="AXTabGroup",
            description="Ana Ruiz, Bei Computeraudio Stummschaltung deaktiviert, Aktiver Sprecher",
        )
        result = _parse.zoom(_root([tile]))
        self.assertIn("Ana Ruiz", result.participants)
        self.assertIn("Ana Ruiz", result.speaking)


# ── Google Meet ───────────────────────────────────────────────────────────────

class TestMeet(unittest.TestCase):
    def test_tile_names(self):
        def _tile(name):
            # 7 levels up: container → ... → name text
            child = _node(role="AXStaticText", title=name)
            n7 = child
            for _ in range(7):
                n7 = _node(children=[n7])
            return n7

        result = _parse.meet(_root([_tile("Ann Lee"), _tile("Bo Park")]))
        self.assertIn("Ann Lee", result.participants)
        self.assertIn("Bo Park", result.participants)

    def test_you_identified(self):
        name = _node(role="AXStaticText", title="You")
        n7 = name
        for _ in range(7):
            n7 = _node(children=[n7])
        result = _parse.meet(_root([n7]))
        self.assertEqual(result.me, "You")

    def test_not_a_name(self):
        for bad in ["00:05", "abc-def-ghi", "123"]:
            name = _node(role="AXStaticText", title=bad)
            n7 = name
            for _ in range(7):
                n7 = _node(children=[n7])
            result = _parse.meet(_root([n7]))
            self.assertNotIn(bad, result.participants)

    def test_empty_tree(self):
        result = _parse.meet(_root([]))
        self.assertEqual(result.participants, [])


# ── Slack Huddle ───────────────────────────────────────────────────────────────

def _huddle_window(*names):
    """A synthetic Slack huddle window: header + one tile per name."""
    children = [_node(role="AXStaticText", title="Video huddle")]
    for n in names:
        children.append(
            {"role": "AXGroup", "description": "", "title": "", "value": "",
             "children": [_node(role="AXStaticText", title=n)]}
        )
    return [
        {"role": "AXWindow", "description": "", "title": "Slack",
         "value": "", "children": children}
    ]


class TestSlack(unittest.TestCase):
    def test_one_to_one(self):
        result = _parse.slack(_huddle_window("Jane Doe", "You"))
        self.assertEqual(result.participants, ["Jane Doe"])
        self.assertEqual(result.me, "You")
        self.assertEqual(result.speaking, [])

    def test_group(self):
        result = _parse.slack(_huddle_window("Jane Doe", "Bo Park", "You"))
        self.assertEqual(sorted(result.participants), ["Bo Park", "Jane Doe"])
        self.assertEqual(result.me, "You")

    def test_ui_words_excluded(self):
        result = _parse.slack(_huddle_window("Jane Doe", "Muted", "End huddle"))
        self.assertEqual(result.participants, ["Jane Doe"])

    def test_chat_window_not_matched(self):
        """Names only count in a root that also carries a huddle header."""
        chat = _huddle_window("Jane Doe")
        chat[0]["children"] = [
            _node(role="AXStaticText", title="Jane Doe"),
            _node(role="AXStaticText", title="Can you take a look at the deploy please"),
        ]  # no huddle header
        self.assertEqual(_parse.slack(chat).participants, [])

    def test_long_chat_line_excluded(self):
        result = _parse.slack(
            _huddle_window(
                "Jane Doe",
                "Can you take a look at the deploy please?",
            )
        )
        self.assertEqual(result.participants, ["Jane Doe"])

    def test_empty_tree(self):
        result = _parse.slack(_root([]))
        self.assertEqual(result.participants, [])
        self.assertIsNone(result.me)


# ── Teams ─────────────────────────────────────────────────────────────────────

class TestTeams(unittest.TestCase):
    def test_attendees_panel(self):
        def _in_list(children):
            return _node(
                role="AXOutline",
                description="Participants",
                children=[
                    _node(
                        role="tree item",
                        description="Dana Smith, Organizer, Unmuted",
                        children=children,
                    ),
                ],
            )

        name = _node(role="AXStaticText", title="Dana Smith, Organizer, Unmuted")
        result = _parse.teams(_root([_in_list([name])]))
        self.assertIn("Dana Smith", result.participants)

    def test_speaking_marker(self):
        result = _parse.teams(
            _root([_node(role="AXStaticText", title="Dana Smith, speaking")])
        )
        self.assertIn("Dana Smith", result.participants)
        self.assertIn("Dana Smith", result.speaking)

    def test_you_marker(self):
        name = _node(role="AXStaticText", title="You (me)")
        result = _parse.teams(_root([_node(role="tree item", description="You (me)", children=[name])]))
        self.assertEqual(result.me, "You")

    def test_empty_tree(self):
        result = _parse.teams(_root([]))
        self.assertEqual(result.participants, [])


# ── Webex ─────────────────────────────────────────────────────────────────────

class TestWebex(unittest.TestCase):
    def test_participant_row(self):
        outline = _node(
            role="AXOutline",
            description="participants",
            children=[
                _node(
                    role="AXRow",
                    description="Ann Lee, speaking, me",
                    children=[_node(role="AXStaticText", title="Ann Lee")],
                ),
            ],
        )
        result = _parse.webex(_root([outline]))
        self.assertIn("Ann Lee", result.participants)
        self.assertIn("Ann Lee", result.speaking)
        self.assertEqual(result.me, "Ann Lee")

    def test_talking_tile(self):
        group = _node(
            role="AXGroup",
            title="Ann Lee, speaking, Video on",
        )
        result = _parse.webex(_root([group]))
        self.assertIn("Ann Lee", result.speaking)

    def test_empty_tree(self):
        result = _parse.webex(_root([]))
        self.assertEqual(result.participants, [])


# ── Dispatch ──────────────────────────────────────────────────────────────────

class TestParse(unittest.TestCase):
    def test_dispatch(self):
        ann = _node(
            role="AXTabGroup",
            description="Ann Lee, Computer audio unmuted",
        )
        self.assertIsInstance(_parse.parse("Zoom", _root([ann])), _parse.CallReading)

    def test_dispatch_slack_huddle(self):
        tree = _huddle_window("Jane Doe", "You")
        result = _parse.parse("Slack Huddle", tree)
        self.assertEqual(result.participants, ["Jane Doe"])

    def test_unknown_app(self):
        result = _parse.parse("UnknownApp", [])
        self.assertEqual(result.participants, [])

    def test_error_tolerant(self):
        result = _parse.parse("Zoom", ["not a dict"])
        self.assertEqual(result.participants, [])


if __name__ == "__main__":
    unittest.main()