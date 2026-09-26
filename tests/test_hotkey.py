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

    def test_caps_lock_is_a_modifier(self):
        key = hotkey.parse("caps+f8")
        self.assertIsNotNone(key)
        self.assertEqual(key.key_code, 99)
        self.assertEqual(key.flags, hotkey.FLAG_CAPS_LOCK)
        self.assertEqual(key.spec, "caps+f8")

    def test_caps_lock_spellings(self):
        for spec, key_code in (
            ("capslock+f8", 99),
            ("caps lock+f8", 99),
            ("\u21ea+t", 17),
            ("CAPS + t", 17),
        ):
            key = hotkey.parse(spec)
            self.assertIsNotNone(key, spec)
            self.assertEqual(key.key_code, key_code)
            self.assertEqual(key.flags, hotkey.FLAG_CAPS_LOCK)

    def test_caps_lock_with_other_modifiers(self):
        key = hotkey.parse("caps+ctrl+alt+d")
        self.assertIsNotNone(key)
        self.assertEqual(
            key.flags,
            hotkey.FLAG_CAPS_LOCK | hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION,
        )
        self.assertEqual(key.spec, "caps+ctrl+alt+d")

    def test_caps_lock_rejected_as_main_key(self):
        # Its down/up pair is reserved for hold tracking; as the main key it
        # would leave the lock state in the user's lap after each dictation.
        self.assertIsNone(hotkey.parse("capslock"))
        self.assertIsNone(hotkey.parse("keycode:57"))
        self.assertIsNone(hotkey.parse("ctrl+keycode:57"))

    def test_bare_caps_lock_is_not_a_combo(self):
        self.assertIsNone(hotkey.parse("caps"))

    def test_keycode_token_requires_digits(self):
        self.assertIsNone(hotkey.parse("keycode:abc"))
        self.assertIsNone(hotkey.parse("keycode:"))
        self.assertIsNone(hotkey.parse("keycode"))

    def test_roundtrip(self):
        for spec in (
            "f8",
            "ctrl+alt+f8",
            "cmd+shift+d",
            "space",
            "ctrl+alt+keycode:70",
            "caps+t",
            "caps+cmd+shift+d",
        ):
            key = hotkey.parse(spec)
            self.assertIsNotNone(key, spec)
            self.assertEqual(hotkey.parse(key.spec), key)


class TestKeyCodesVerified(unittest.TestCase):
    """Pin the letter/digit codes verified against the OS's compiled layout.

    These are NOT the values of the kVK_ANSI_* table circulating in old
    headers and blog posts — see the KEY_CODES comment in trnscrb.hotkey.
    """

    def test_verified_codes(self):
        for name, code in {
            "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5,
            "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
            "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
            "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
            "9": 25, "7": 26, "8": 28, "0": 29,
            "o": 31, "u": 32, "i": 34, "p": 35,
            "l": 37, "j": 38, "k": 40,
        }.items():
            self.assertEqual(hotkey.KEY_CODES[name], code, name)


class TestCanonicalAndDisplay(unittest.TestCase):
    def test_canonical_spec_orders_modifiers(self):
        # Canonical order is ⇪⌃⌥⇧⌘ regardless of the order typed.
        self.assertEqual(
            hotkey.canonical_spec("d", hotkey.FLAG_COMMAND | hotkey.FLAG_CONTROL),
            "ctrl+cmd+d",
        )
        self.assertEqual(
            hotkey.canonical_spec("t", hotkey.FLAG_CAPS_LOCK | hotkey.FLAG_COMMAND),
            "caps+cmd+t",
        )

    def test_display_spec(self):
        self.assertEqual(
            hotkey.display_spec(99, hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION),
            "\u2303\u2325F8",
        )
        self.assertEqual(hotkey.display_spec(0, 0), "A")
        self.assertEqual(hotkey.display_spec(49, hotkey.FLAG_COMMAND), "\u2318Space")
        self.assertEqual(hotkey.display_spec(70, hotkey.FLAG_SHIFT), "\u21e7key 70")
        self.assertEqual(hotkey.display_spec(17, hotkey.FLAG_CAPS_LOCK), "\u21eaT")
        self.assertEqual(
            hotkey.display_spec(
                99,
                hotkey.FLAG_CAPS_LOCK | hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION,
            ),
            "\u21ea\u2303\u2325F8",
        )

    def test_ptt_mods_drops_shift(self):
        self.assertEqual(
            hotkey.ptt_mods(hotkey.FLAG_SHIFT | hotkey.FLAG_CONTROL),
            hotkey.FLAG_CONTROL,
        )
        self.assertEqual(hotkey.ptt_mods(0), 0)

    def test_ptt_mods_keeps_caps_lock(self):
        # Caps lock counts toward a combo (unlike shift): record mode must
        # be able to capture it.
        self.assertEqual(
            hotkey.ptt_mods(hotkey.FLAG_CAPS_LOCK | hotkey.FLAG_SHIFT),
            hotkey.FLAG_CAPS_LOCK,
        )

    def test_key_name_reverse_lookup(self):
        self.assertEqual(hotkey.key_name(99), "f8")
        self.assertIsNone(hotkey.key_name(70))


if __name__ == "__main__":
    unittest.main()
