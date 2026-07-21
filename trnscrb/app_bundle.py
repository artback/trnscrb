"""Wrapper app bundle so macOS permission prompts say "Trnscrb".

TCC attributes permissions to the *responsible process*. Launched from a
terminal, that's the terminal app (prompts say "Ghostty"/"Terminal"); launched
bare from launchd, it's the naked python binary. Wrapping the launch in a
minimal ``Trnscrb.app`` — whose main executable is a tiny compiled launcher
that spawns ``trnscrb start`` as a child and stays resident — makes the app
bundle the responsible process, so Screen Recording, Microphone, and
Automation prompts are attributed to "Trnscrb" and survive updates.

The launcher is compiled with the system C compiler (present wherever
Homebrew is, since brew requires the Xcode CLT); if no compiler is available
we fall back to a shell-script launcher, which still works but macOS may
attribute prompts less cleanly.
"""

import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.app_bundle")

BUNDLE_ID = "io.trnscrb.app"
_LAUNCHER_VERSION = 1  # bump to force a rebuild on upgrade

_LAUNCHER_C = """\
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {{
    if (child > 0) kill(child, sig);
}}

int main(void) {{
    char *argv[] = {{"{target}", "start", NULL}};
    setenv("TRNSCRB_IN_BUNDLE", "1", 1);
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    if (posix_spawn(&child, "{target}", NULL, NULL, argv, environ) != 0)
        return 1;
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {{
        /* interrupted by a forwarded signal — keep waiting */
    }}
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return 128;
}}
"""

_LAUNCHER_SH = """\
#!/bin/sh
# Fallback launcher (no C compiler available at install time).
export TRNSCRB_IN_BUNDLE=1
exec "{target}" start
"""


def bundle_path() -> Path:
    return Path.home() / "Applications" / "Trnscrb.app"


def executable_path() -> Path:
    return bundle_path() / "Contents" / "MacOS" / "Trnscrb"


def is_current(trnscrb_binary: str) -> bool:
    """True if the installed bundle already wraps this binary at this version."""
    marker = bundle_path() / "Contents" / "Resources" / "launcher.txt"
    try:
        return marker.read_text() == _marker(trnscrb_binary)
    except OSError:
        return False


def _marker(trnscrb_binary: str) -> str:
    return f"{_LAUNCHER_VERSION}\n{trnscrb_binary}\n"


def _info_plist(version: str) -> dict:
    return {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": "Trnscrb",
        "CFBundleDisplayName": "Trnscrb",
        "CFBundleExecutable": "Trnscrb",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "13.0",
        # Menu bar app — no Dock icon, no app switcher entry.
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": (
            "Trnscrb records your microphone during meetings to transcribe them."
        ),
        "NSAppleEventsUsageDescription": (
            "Trnscrb checks browsers and meeting apps to detect when a meeting starts and ends."
        ),
        "NSCalendarsUsageDescription": (
            "Trnscrb reads your calendar to name transcripts after the meeting."
        ),
    }


def _build_launcher(target: str, dest: Path) -> str:
    """Create the bundle's main executable. Returns 'compiled' or 'script'."""
    cc = shutil.which("cc") or shutil.which("clang")
    if cc:
        src = dest.parent / "launcher.c"
        src.write_text(_LAUNCHER_C.format(target=target))
        try:
            subprocess.run(
                [cc, "-O2", "-o", str(dest), str(src)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            return "compiled"
        except Exception as e:
            _log.warning("Launcher compile failed (%s); using script fallback", e)
        finally:
            src.unlink(missing_ok=True)
    dest.write_text(_LAUNCHER_SH.format(target=target))
    dest.chmod(0o755)
    return "script"


def _codesign(bundle: Path) -> None:
    """Ad-hoc sign so the TCC identity stays stable across rebuilds."""
    codesign = shutil.which("codesign")
    if not codesign:
        return
    try:
        subprocess.run(
            [codesign, "--force", "--deep", "--sign", "-", str(bundle)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except Exception as e:
        _log.warning("Ad-hoc codesign failed (harmless, but TCC may re-prompt): %s", e)


def build_bundle(dest: Path, trnscrb_binary: str) -> Path | None:
    """Assemble a Trnscrb.app at ``dest`` wrapping ``trnscrb_binary``.

    Used both by ``ensure_bundle`` (fallback local build) and by the Homebrew
    formula (``python -m trnscrb.app_bundle <dest> <target>``), which ships
    the resulting bundle inside the package. Returns the executable path.
    """
    try:
        from trnscrb import __version__

        contents = dest / "Contents"
        macos_dir = contents / "MacOS"
        resources = contents / "Resources"
        macos_dir.mkdir(parents=True, exist_ok=True)
        resources.mkdir(parents=True, exist_ok=True)

        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump(_info_plist(__version__), f)

        executable = macos_dir / "Trnscrb"
        executable.unlink(missing_ok=True)
        kind = _build_launcher(trnscrb_binary, executable)
        _codesign(dest)
        (resources / "launcher.txt").write_text(_marker(trnscrb_binary))
        _log.info("App bundle ready at %s (%s launcher)", dest, kind)
        return executable
    except Exception:
        _log.warning("App bundle creation failed", exc_info=True)
        return None


def _bundle_version(bundle: Path) -> str | None:
    try:
        with open(bundle / "Contents" / "Info.plist", "rb") as f:
            return str(plistlib.load(f).get("CFBundleVersion") or "") or None
    except Exception:
        return None


def _packaged_bundle(binary: Path) -> Path | None:
    """Find a Trnscrb.app shipped inside the package prefix (e.g. by Homebrew)."""
    try:
        real = binary.resolve()
        installed = bundle_path().resolve()
    except OSError:
        return None
    for parent in list(real.parents)[:5]:
        candidate = parent / "Trnscrb.app"
        if candidate == installed:
            continue
        if (candidate / "Contents" / "MacOS" / "Trnscrb").exists():
            return candidate
    return None


def _launchers_identical(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _install_packaged(packaged: Path) -> Path | None:
    """Copy the package-shipped bundle into ~/Applications when needed.

    TCC ties Screen Recording grants to the bundle's (ad-hoc) code
    signature, and every copy is a fresh signature — so the installed bundle
    is REPLACED ONLY when its launcher binary actually differs. The launcher
    targets the version-stable opt path, so routine version bumps leave it
    byte-identical and the user's permission grant intact (Info.plist keeps
    the older version string; that is deliberate and purely cosmetic).
    """
    installed = bundle_path()
    executable = executable_path()
    try:
        if executable.exists() and _launchers_identical(
            executable, packaged / "Contents" / "MacOS" / "Trnscrb"
        ):
            return executable
        replacing = installed.exists()
        if replacing:
            shutil.rmtree(installed)
        installed.parent.mkdir(parents=True, exist_ok=True)
        # ditto preserves the code signature; copytree is the fallback
        result = subprocess.run(
            ["ditto", str(packaged), str(installed)], capture_output=True, timeout=60
        )
        if result.returncode != 0:
            shutil.copytree(packaged, installed, symlinks=True)
        if replacing:
            _log.warning(
                "App bundle launcher changed — the Screen Recording permission "
                "must be granted again (System Settings, or approve the prompt "
                "at the next recording)."
            )
        _log.info("Installed packaged app bundle from %s", packaged)
        return executable
    except Exception:
        _log.warning("Could not install packaged bundle from %s", packaged, exc_info=True)
        return None


def is_installed(trnscrb_binary: str) -> bool:
    """True if ~/Applications/Trnscrb.app is present and up to date for this binary.

    "Up to date" means the launcher binary matches — version strings are
    ignored on purpose, since replacing the bundle invalidates the user's
    Screen Recording grant (TCC is keyed to the code signature).
    """
    if is_current(trnscrb_binary):
        return True
    packaged = _packaged_bundle(Path(trnscrb_binary))
    if packaged is None or not executable_path().exists():
        return False
    return _launchers_identical(executable_path(), packaged / "Contents" / "MacOS" / "Trnscrb")


def ensure_bundle(trnscrb_binary: str | None = None) -> Path | None:
    """Install or refresh ~/Applications/Trnscrb.app. Returns the executable path.

    Prefers the prebuilt bundle shipped inside the package (Homebrew builds it
    at package-install time) and just copies it; falls back to building one
    locally for pip/uv installs. Idempotent. None if neither path worked.
    """
    trnscrb_binary = trnscrb_binary or shutil.which("trnscrb") or sys.argv[0]
    if not trnscrb_binary or not Path(trnscrb_binary).exists():
        _log.warning("Cannot set up app bundle: trnscrb binary not found")
        return None

    packaged = _packaged_bundle(Path(trnscrb_binary))
    if packaged is not None:
        executable = _install_packaged(packaged)
        if executable is not None:
            return executable

    if is_current(trnscrb_binary):
        return executable_path()
    return build_bundle(bundle_path(), trnscrb_binary)


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m trnscrb.app_bundle <dest-bundle> <target-binary>",
            file=sys.stderr,
        )
        return 2
    return 0 if build_bundle(Path(argv[0]), argv[1]) else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
