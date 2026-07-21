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

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.app_bundle")

BUNDLE_ID = "io.trnscrb.app"
# Bundle identity version: bump ONLY for changes that must reach installed
# bundles (launcher behavior, icon, plist capabilities). Each bump replaces
# the installed bundle → new ad-hoc signature → the user must re-grant
# Screen Recording once. Routine releases must NOT bump this.
_LAUNCHER_VERSION = 4  # v4: embedded Python so TCC sees the bundle's identity

_LAUNCHER_C = """\
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {{
    if (child > 0) kill(child, sig);
}}

int main(void) {{
    /* Run the Python copy that lives INSIDE this bundle, so the process that
       calls ScreenCaptureKit carries the bundle's code identity (see the
       module docstring). Resolve it relative to ourselves — the bundle is
       built in the package prefix and copied to ~/Applications. */
    char python[PATH_MAX];
    uint32_t size = sizeof(python);
    if (_NSGetExecutablePath(python, &size) != 0)
        return 1;
    char *slash = strrchr(python, '/');
    if (slash == NULL || (slash - python) + sizeof("{python_name}") >= PATH_MAX)
        return 1;
    strcpy(slash + 1, "{python_name}");

    setenv("TRNSCRB_IN_BUNDLE", "1", 1);
    setenv("PYTHONHOME", "{pythonhome}", 1);
    setenv("PYTHONPATH", "{pythonpath}", 1);
    setenv("PYTHONNOUSERSITE", "1", 1);

    char *argv[] = {{python, "{target}", "start", NULL}};
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    if (posix_spawn(&child, python, NULL, NULL, argv, environ) != 0)
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
# Note: a shell launcher makes /bin/sh the running binary, so macOS will not
# attribute Screen Recording to this bundle — the compiled launcher is what
# gives the app its own TCC identity.
export TRNSCRB_IN_BUNDLE=1
export PYTHONHOME="{pythonhome}"
export PYTHONPATH="{pythonpath}"
export PYTHONNOUSERSITE=1
exec "$(dirname "$0")/{python_name}" "{target}" start
"""

_EMBEDDED_PYTHON = "TrnscrbPython"  # Python copy inside Contents/MacOS/


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


def _info_plist(version: str, has_icon: bool = False) -> dict:
    plist = {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": "Trnscrb",
        "CFBundleDisplayName": "Trnscrb",
        "CFBundleExecutable": "Trnscrb",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHumanReadableCopyright": "MIT License",
        "NSHighResolutionCapable": True,
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
    if has_icon:
        plist["CFBundleIconFile"] = "Trnscrb"
    return plist


def _stable_path(path: str) -> str:
    """Rewrite a versioned Homebrew Cellar path to its stable opt equivalent.

    /opt/homebrew/Cellar/trnscrb/1.2.3/x → /opt/homebrew/opt/trnscrb/x
    Baking a Cellar path into the launcher would change its bytes every
    release, replacing the bundle and voiding the user's TCC grant.
    """
    parts = Path(path).parts
    try:
        i = parts.index("Cellar")
    except ValueError:
        return path
    if len(parts) < i + 3:
        return path
    candidate = Path(*parts[:i], "opt", parts[i + 1], *parts[i + 3 :])
    return str(candidate) if candidate.exists() else path


def _python_runtime() -> tuple[str, str, str]:
    """(python_binary, PYTHONHOME, PYTHONPATH) for the embedded interpreter."""
    import sysconfig

    python_binary = os.path.realpath(sys.executable)
    home = sys.base_prefix
    purelib = _stable_path(sysconfig.get_paths()["purelib"])
    return python_binary, home, purelib


def _python_script(target: str) -> str:
    """The Python script the embedded interpreter should run.

    The launcher invokes ``TrnscrbPython <script> start``, so the script must
    be Python. Homebrew's ``bin/trnscrb`` is a *shell* wrapper (it sets PATH
    and execs the venv script), which Python cannot parse — in that case use
    the venv console script this build is running from.
    """
    try:
        first_line = Path(target).read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError):
        first_line = ""
    if first_line.startswith("#!") and "python" not in first_line:
        derived = _stable_path(os.path.join(sys.prefix, "bin", "trnscrb"))
        if Path(derived).exists():
            _log.debug("Target %s is a shell wrapper; launching %s", target, derived)
            return derived
    return target


def _build_launcher(target: str, dest: Path) -> str:
    """Create the bundle's main executable. Returns 'compiled' or 'script'."""
    _python_bin, pythonhome, pythonpath = _python_runtime()
    fmt = {
        "target": _python_script(target),
        "pythonhome": pythonhome,
        "pythonpath": pythonpath,
        "python_name": _EMBEDDED_PYTHON,
    }
    cc = shutil.which("cc") or shutil.which("clang")
    if cc:
        src = dest.parent / "launcher.c"
        src.write_text(_LAUNCHER_C.format(**fmt))
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
    dest.write_text(_LAUNCHER_SH.format(**fmt))
    dest.chmod(0o755)
    return "script"


def _codesign(bundle: Path) -> None:
    """Ad-hoc sign the finished bundle so TCC can persist grants for it.

    Must be the LAST build step — codesign seals the current bundle contents,
    and any later change invalidates the seal. The resulting cdhash is stable
    across rebuilds as long as the sealed bytes are identical, which is what
    lets a Screen Recording grant survive upgrades.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        return
    try:
        # --identifier is essential: the embedded Python binary carries its own
        # __info_plist declaring org.python.python, which codesign would
        # otherwise adopt — and TCC would attribute recordings to "Python"
        # instead of Trnscrb, so the user's grant would never apply.
        embedded = bundle / "Contents" / "MacOS" / _EMBEDDED_PYTHON
        if embedded.exists():
            subprocess.run(
                [codesign, "--force", "--identifier", BUNDLE_ID, "--sign", "-", str(embedded)],
                check=True,
                capture_output=True,
                timeout=60,
            )
        subprocess.run(
            [codesign, "--force", "--identifier", BUNDLE_ID, "--sign", "-", str(bundle)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        verify = subprocess.run(
            [codesign, "--verify", "--strict", str(bundle)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if verify.returncode != 0:
            _log.warning("Bundle signature verification failed: %s", verify.stderr.strip())
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

        from trnscrb.app_icon import build_icns

        has_icon = build_icns(resources / "Trnscrb.icns")

        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump(_info_plist(__version__, has_icon=has_icon), f)

        # Copy the interpreter INTO the bundle — a process running Homebrew's
        # Python.app is identified by macOS as org.python.python, so a grant
        # given to Trnscrb would never apply to it.
        python_binary, _home, _path = _python_runtime()
        embedded_python = macos_dir / _EMBEDDED_PYTHON
        embedded_python.unlink(missing_ok=True)
        shutil.copy2(python_binary, embedded_python)
        embedded_python.chmod(0o755)

        executable = macos_dir / "Trnscrb"
        executable.unlink(missing_ok=True)
        kind = _build_launcher(trnscrb_binary, executable)
        # Every file must be in place BEFORE signing — codesign seals the
        # bundle's resources, and adding a file afterward invalidates the seal
        # ("a sealed resource is missing or invalid"), which stops macOS from
        # persisting the Screen Recording grant across launches.
        (resources / "launcher.txt").write_text(_marker(trnscrb_binary))
        _codesign(dest)
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


def _bundle_marker(bundle: Path) -> str | None:
    """Identity marker of a bundle (launcher version + wrapped target)."""
    try:
        return (bundle / "Contents" / "Resources" / "launcher.txt").read_text()
    except OSError:
        return None


def _install_packaged(packaged: Path) -> Path | None:
    """Copy the package-shipped bundle into ~/Applications when needed.

    TCC ties Screen Recording grants to the bundle's (ad-hoc) code
    signature, and every copy is a fresh signature — so the installed bundle
    is REPLACED ONLY when its identity marker (launcher version + wrapped
    target) differs. Routine version bumps keep the marker identical and the
    user's permission grant intact (Info.plist keeps the older version
    string; that is deliberate and purely cosmetic). Bumping
    _LAUNCHER_VERSION is the explicit, documented way to roll the identity.
    """
    installed = bundle_path()
    executable = executable_path()
    try:
        installed_marker = _bundle_marker(installed)
        if (
            executable.exists()
            and installed_marker is not None
            and installed_marker == _bundle_marker(packaged)
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

    "Up to date" means the identity marker matches — version strings are
    ignored on purpose, since replacing the bundle invalidates the user's
    Screen Recording grant (TCC is keyed to the code signature).
    """
    if is_current(trnscrb_binary):
        return True
    packaged = _packaged_bundle(Path(trnscrb_binary))
    if packaged is None or not executable_path().exists():
        return False
    installed_marker = _bundle_marker(bundle_path())
    return installed_marker is not None and installed_marker == _bundle_marker(packaged)


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
