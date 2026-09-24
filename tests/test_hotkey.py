"""Tests for PTT key-spec parsing, canonicalization, and display.

``trnscrb.hotkey`` is pure (no PyObjC), so these run on any platform.
"""

import unittest

from trnscrb import hotkey


class TestParse(unittest.TestCase):
    def test_bare_key(self):
        key = hotkey.parse("f8")
        self.assertIsNotNone(key)
        self.assertEqual(key.key_code, 99)
        self.assertEqual(key.flags, 0)
        self.assertEqual(key.spec, "f8")

    def test_letter_with_modifiers(self):
        key = hotkey.parse("ctrl+alt+d")
        self.assertEqual(key.key_code, 2)
        self.assertEqual(key.flags, hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION)
        self.assertEqual(key.spec, "ctrl+alt+d")

    def test_case_and_space_insensitive(self):
        key = hotkey.parse("  CMD + SHIFT + F2  ")
        self.assertIsNotNone(key)
        self.assertEqual(key.key_code, 97)
        self.assertEqual(key.flags, hotkey.FLAG_COMMAND | hotkey.FLAG_SHIFT)

    def test_modifier_aliases(self):
        key = hotkey.parse("command+option+d")
        self.assertEqual(key.flags, hotkey.FLAG_COMMAND | hotkey.FLAG_OPTION)

    def test_unicode_modifier_names(self):
        key = hotkey.parse("\u2303+\u2325+f8")  # ⌃⌥f8
        self.assertIsNotNone(key)
        self.assertEqual(key.spec, "ctrl+alt+f8")

    def test_modifiers_only_is_not_a_combo(self):
        self.assertIsNone(hotkey.parse("ctrl"))
        self.assertIsNone(hotkey.parse("ctrl+alt"))

    def test_unknown_key(self):
        self.assertIsNone(hotkey.parse("ctrl+alt+q2"))
        self.assertIsNone(hotkey.parse("warp9"))

    def test_empty_is_off(self):
        self.assertIsNone(hotkey.parse(""))
        self.assertIsNone(hotkey.parse("   "))
        self.assertIsNone(hotkey.parse(None))

    def test_raw_keycode_token(self):
        # Record mode stores combos it cannot name as ``keycode:N``.
        key = hotkey.parse("ctrl+alt+keycode:45")
        self.assertEqual(key.key_code, 45)
        self.assertEqual(key.flags, hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION)
        self.assertEqual(key.spec, "ctrl+alt+keycode:45")

    def test_keycode_token_requires_digits(self):
        self.assertIsNone(hotkey.parse("keycode:abc"))
        self.assertIsNone(hotkey.parse("keycode:"))
        self.assertIsNone(hotkey.parse("keycode"))

    def test_roundtrip(self):
        for spec in ("f8", "ctrl+alt+f8", "cmd+shift+d", "space", "ctrl+alt+keycode:70"):
            key = hotkey.parse(spec)
            self.assertIsNotNone(key, spec)
            self.assertEqual(hotkey.parse(key.spec), key)


class TestCanonicalAndDisplay(unittest.TestCase):
    def test_canonical_spec_orders_modifiers(self):
        # Canonical order is ⌃⌥⇧⌘ regardless of the order typed.
        self.assertEqual(
            hotkey.canonical_spec("d", hotkey.FLAG_COMMAND | hotkey.FLAG_CONTROL),
            "ctrl+cmd+d",
        )

    def test_display_spec(self):
        self.assertEqual(
            hotkey.display_spec(99, hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION),
            "\u2303\u2325F8",
        )
        self.assertEqual(hotkey.display_spec(0, 0), "A")
        self.assertEqual(hotkey.display_spec(49, hotkey.FLAG_COMMAND), "\u2318Space")
        self.assertEqual(hotkey.display_spec(70, hotkey.FLAG_SHIFT), "\u21e7key 70")

    def test_ptt_mods_drops_shift(self):
        self.assertEqual(
            hotkey.ptt_mods(hotkey.FLAG_SHIFT | hotkey.FLAG_CONTROL),
            hotkey.FLAG_CONTROL,
        )
        self.assertEqual(hotkey.ptt_mods(0), 0)

    def test_key_name_reverse_lookup(self):
        self.assertEqual(hotkey.key_name(99), "f8")
        self.assertIsNone(hotkey.key_name(70))


if __name__ == "__main__":
    unittest.main()
