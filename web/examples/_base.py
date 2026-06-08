"""Base types for the antenna-example registry.

An `AntennaExample` bundles everything the web layer needs to serve one
geometry: parameter schema, pysim solve/sweep, and pynec build/solve. Any
callable left as None signals the backend doesn't support that operation
for this geometry.

Keeping these as plain callables (rather than a class hierarchy) means
each example module is a flat file of functions plus one EXAMPLE = ...
assignment at the bottom — easy to read, easy to delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

# A pynec "build" returns the dict-of-context-plus-derived-geom that both
# solve() and pattern() consume — see web/pynec_backend.py for the shape.
PynecBuildFn = Callable[[dict], dict]

# pysim solve / pynec solve: take a request dict, return the response dict
# the frontend renders.
SolveFn = Callable[[dict], dict]

# pysim sweep: take a request dict + frequency list. Single-feed geometries
# return (re, im) — two parallel lists of input impedance. Multi-feed
# geometries (bowtie 1×2) return (primary_re, primary_im, feeds_re,
# feeds_im) where feeds_* is (n_freqs × n_feeds). The sweep endpoint
# detects the tuple length to know which shape it got.
SweepResult = Union[
    Tuple[list[float], list[float]],
    Tuple[list[float], list[float], list[list[float]], list[list[float]]],
]
SweepFn = Callable[[dict, list[float]], SweepResult]

# pynec pattern excitation: drive the NEC context using the build dict's
# feed metadata. Single-feed default lives in web.pynec_backend.pattern();
# multi-feed examples (bowtie) supply their own (e.g. two ex_card calls).
PynecPatternExciteFn = Callable[[dict, float], None]  # (build, freq_mhz)


@dataclass(frozen=True)
class ParamSpec:
    """One UI-exposed parameter for a geometry.

    Served from `GET /examples` so the frontend can render parameter
    controls generically — one ParamForm component reads the schema and
    builds the matching <input> elements without per-antenna JSX.

    `visible_when` makes a parameter conditional on another's value:
    e.g. yagi's director_spacing_wavelengths slider only appears when
    n_directors > 0. The dict is shaped {name, op, value}; ops include
    "eq", "ne", "gt", "ge", "lt", "le".

    `kind` switches the input type. "int" renders an integer step slider,
    "float" a continuous slider, "bool" a checkbox. Bowtie's phase_lr_deg
    uses a signed range — that's still kind="float" with min/max < 0.
    """

    name: str
    label: str
    default: Any
    kind: str = "float"  # float | int | bool
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    precision: int = 3  # decimal places shown in the value readout
    unit: Optional[str] = None  # rendered next to the value: "°", "m", "λ", etc.
    visible_when: Optional[dict] = None  # {"name": ..., "op": ..., "value": ...}
    sweepable: bool = False


@dataclass(frozen=True)
class AntennaExample:
    name: str
    label: str
    pysim_solve: SolveFn
    pysim_sweep: SweepFn
    pynec_build: Optional[PynecBuildFn] = None
    pynec_solve: Optional[SolveFn] = None
    # When set, pattern() calls this to excite the NEC context (e.g.
    # multi-source drive for bowtie's 1×2 array). When None, pattern()
    # uses the default single-feed excitation reading b["feed_seg"] /
    # b["feed_tag"] / b["n_per_wire"] from the build dict.
    pynec_pattern_excite: Optional[PynecPatternExciteFn] = None
    # Sweep returns the multi-feed 4-tuple instead of the single-feed
    # 2-tuple. Sweep endpoint uses this to dispatch the two response
    # streaming shapes.
    multi_feed: bool = False
    # When True, the frontend falls through to hardcoded JSX controls
    # instead of generic ParamForm rendering. Used by fan_dipole, whose
    # per-band UI (list of bands, each with its own selectors and sliders)
    # doesn't fit a flat ParamSpec list. Schema-driven examples set it
    # to False and supply a full param_schema.
    legacy_controls: bool = False
    param_schema: tuple[ParamSpec, ...] = field(default_factory=tuple)
