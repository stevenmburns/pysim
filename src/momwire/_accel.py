"""Single load point for the optional C++ accelerator (``_accelerators``).

Every solver module imports the extension through here instead of carrying its
own ``try/except ImportError`` guard, so the decision of *whether to warn* lives
in one place. The distinction that matters:

* **Extension never built** (unsupported platform, or a deliberate pure-Python
  install) — the pure-Python fallback is expected, so stay silent.
* **Extension built but failed to load** — something is wrong at *runtime*, and
  the fast path silently vanishes, so warn loudly. The classic cause on Linux is
  a static-TLS clash when another OpenMP extension is ``dlopen``'d first (e.g. a
  pynec-accel build that vendors its own ``libgomp``); the import then fails with
  "cannot allocate memory in static TLS block" and every solver quietly drops to
  the slow path. That used to be invisible — this module makes it audible.

Public attributes:
    ``acc``     — the loaded ``_accelerators`` module, or ``None``.
    ``LOADED``  — ``True`` iff the accelerator imported successfully.
"""

from __future__ import annotations

import importlib.machinery
import pathlib
import warnings


def _extension_built() -> bool:
    """True if a compiled ``_accelerators`` extension exists on disk.

    Distinguishes "built but won't load" from "never built": the file's presence
    means the build succeeded, so a failed import is a runtime problem worth a
    warning rather than an expected pure-Python fallback.
    """
    pkg = pathlib.Path(__file__).parent
    return any(
        (pkg / f"_accelerators{suffix}").exists()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _load():
    """Import the accelerator, warning if a *built* extension fails to load.

    Returns ``(module_or_None, loaded_bool)``. Kept as a function so the warn
    decision is unit-testable without reloading the whole package.
    """
    try:
        from . import _accelerators as mod
    except ImportError as exc:
        if _extension_built():
            warnings.warn(
                "momwire: the compiled accelerator '_accelerators' is installed "
                f"but failed to import ({exc!r}); falling back to the slower "
                "pure-Python path. On Linux this is usually a static-TLS clash "
                "with another OpenMP extension loaded first — e.g. a pynec-accel "
                "build that vendors its own libgomp (upgrade to pynec-accel "
                ">= 1.7.4.post1). Stopgap: "
                "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152 (set "
                "before the interpreter starts).",
                RuntimeWarning,
                stacklevel=3,
            )
        # else: genuinely not built for this platform — pure-Python is expected.
        return None, False
    return mod, True


acc, LOADED = _load()
