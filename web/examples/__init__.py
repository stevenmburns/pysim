"""Antenna example registry.

Each antenna geometry exposed by the web UI is defined in its own module
under this package and registered via `register(EXAMPLE)`. The dispatchers
in `web.server` and `web.pynec_backend` look up the example by its `name`
field, so removing an antenna is a two-line change: delete the module file
and the matching `from . import <name>` line below.

The registry is intentionally web-layer (not solver-core) — each example
parses the request dict, calls shared helpers in `web.server` /
`web.pynec_backend`, and produces the JSON-shaped response the frontend
consumes. The solver package `momwire/` stays free of UI concerns.
"""

from __future__ import annotations

from ._base import AntennaExample, ParamSpec

REGISTRY: dict[str, AntennaExample] = {}


def register(example: AntennaExample) -> AntennaExample:
    if example.name in REGISTRY:
        raise ValueError(f"duplicate antenna example: {example.name}")
    REGISTRY[example.name] = example
    return example


# Importing each example module triggers its register() call. Adding a new
# antenna = create the module + add one import line; removing one = delete
# the module file + the import line.
from . import bowtie_1x2  # noqa: F401,E402
from . import fan_dipole  # noqa: F401,E402
from . import hentenna  # noqa: F401,E402
from . import hexbeam  # noqa: F401,E402
from . import hexbeam_5band  # noqa: F401,E402
from . import inverted_v  # noqa: F401,E402
from . import moxon  # noqa: F401,E402
from . import yagi  # noqa: F401,E402

__all__ = ["AntennaExample", "ParamSpec", "REGISTRY", "register"]
