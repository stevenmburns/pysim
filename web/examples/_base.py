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

    `kind` switches the input type:
      - "float": continuous slider (use min/max/step)
      - "int": integer slider (use min/max/step=1)
      - "bool": checkbox
      - "enum": <select> dropdown; supply `enum_options`
    Bowtie's phase_lr_deg uses a signed range — kind="float" with min<0.

    For "enum" params, `enum_options` is a tuple of free-form dicts. Each
    dict must have at least {"value", "label"}; antennas can attach
    arbitrary extra keys to drive related controls. Fan_dipole's per-band
    `band_id` carries {"value": "20m", "label": "20m", "freq_min": 14.0,
    "freq_max": 14.35, "freq_default": 14.175} so a sibling freq slider
    can read its bounds + default off the active band.

    `range_from_enum_option` lets a slider's min/max/step come from the
    currently-selected enum value of a sibling param. Shape:
    {"param": <enum_param_name>, "min_key": <key>, "max_key": <key>}.

    `on_change_set` is a side-effect rule: when this enum changes value,
    write a sibling param using a key from the new enum option. Shape:
    {"set": <sibling_name>, "from_enum_key": <key>}. Fan_dipole's
    band_id uses {"set": "freq", "from_enum_key": "freq_default"} so
    flipping the band pulldown snaps the freq slider to that band's
    centre.

    `linked_to_design_freq` means this param's value also drives the
    global designFreq state on the frontend. Fan_dipole uses it on
    `bands[0].freq` so the first band's frequency stays the antenna's
    design frequency without a separate global slider.
    """

    name: str
    label: str
    default: Any
    kind: str = "float"  # float | int | bool | enum
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    precision: int = 3  # decimal places shown in the value readout
    unit: Optional[str] = None  # rendered next to the value: "°", "m", "λ", etc.
    visible_when: Optional[dict] = None  # {"name": ..., "op": ..., "value": ...}
    enum_options: Optional[tuple[dict, ...]] = None
    range_from_enum_option: Optional[dict] = None
    on_change_set: Optional[dict] = None
    linked_to_design_freq: bool = False
    sweepable: bool = False


@dataclass(frozen=True)
class ParamGroupSpec:
    """A repeating section of params, count-controlled by another param.

    Used for multi-band / multi-element antennas where each band (or
    element) carries its own set of knobs. Fan_dipole is the first user
    (one group per band: band_id + freq + length_factor). The pattern
    generalises: a trapped multi-bander would have a group with
    {band_id, trap_freq, trap_q}; a multi-band Yagi would have a group
    with {driver_factor, reflector_factor, n_directors, ...} possibly
    containing its own nested director sub-group.

    The schema currently allows nested groups (params is a union type)
    but the frontend only renders one level of nesting in the first
    cut. Adding deeper nesting is a frontend-only change.

    `label_template` is the heading shown above each instance with
    {i} substituted to the 0-indexed instance number, e.g. "band {i}".

    `repeat_count` names a scalar param (kind="int") elsewhere in the
    same schema whose value determines how many instances render.
    `max_repeats` is the absolute cap used for state pre-allocation —
    matching the count param's `max`.
    """

    name: str  # key under which the frontend stores per-instance values
    label_template: str  # e.g. "band {i}" — {i} is the 0-indexed instance
    repeat_count: str  # name of the scalar param that controls how many
    max_repeats: int
    params: tuple[Any, ...]  # tuple[Union[ParamSpec, ParamGroupSpec], ...]
    # Optional per-instance default overrides. Index i applies to
    # instance i; each dict maps {param_name: default_value} and beats
    # the ParamSpec.default for that one instance. Used by fan_dipole
    # so the 5 max bands seed to (20m, 10m, 17m, 12m, 15m) instead of
    # all defaulting to the same band. Tuple length doesn't need to
    # match max_repeats; missing indices fall back to the ParamSpec
    # default.
    default_overrides: tuple[dict, ...] = ()
    # When set, names a param inside `params` (e.g. "freq") whose
    # current per-instance value drives the global measFreq state
    # *every time any leaf in the touched instance changes*. UX
    # rationale: tuning any knob of band i (length factor, t0 factor,
    # ...) signals the user is focused on band i, so the live
    # simulation should track that band's freq. Distinct from
    # ParamSpec.linked_to_design_freq, which is a steady-state link
    # to a single fixed instance (always band 0). This is a
    # per-interaction "follow the user's focus" link, fan_dipole and
    # hexbeam_5band both use "freq".
    # Gated by the frontend's linkMeas toggle.
    link_meas_freq_to_param: Optional[str] = None


@dataclass(frozen=True)
class BandSpec:
    """A frequency-preset tab the UI offers as a design-frequency selector.

    The solver only sees the resulting `design_freq_mhz` float; bands are
    purely a UI affordance. Examples that target HF amateur bands reuse
    `DEFAULT_HF_BANDS`; others can supply their own list, or set bands=()
    to suppress the row entirely (fan_dipole does this — its per-band
    schema-driven controls own the design frequency).
    """

    key: str        # stable identifier; also used as the visible tab label today
    label: str
    freq_mhz: float  # tab default — slider snaps here when the band is selected
    min_mhz: float   # slider lower bound while this band is active
    max_mhz: float


DEFAULT_HF_BANDS: tuple[BandSpec, ...] = (
    BandSpec("20m", "20m", 14.300, 14.000, 14.350),
    BandSpec("17m", "17m", 18.1575, 18.068, 18.168),
    BandSpec("15m", "15m", 21.383, 21.000, 21.450),
    BandSpec("12m", "12m", 24.970, 24.890, 24.990),
    BandSpec("10m", "10m", 28.470, 28.000, 29.700),
)


@dataclass(frozen=True)
class ResultFieldSpec:
    """One field from the solve response to surface in the result panel.

    The frontend's ResultPanel reads the field by name off the response,
    formats it with `precision`, appends `unit`, and renders one row
    labelled `label`. Scalar number fields only — multi-feed tables
    stay hardcoded for now.
    """

    field: str  # response key, e.g. "arm_len_m"
    label: str
    precision: int = 3
    unit: Optional[str] = None  # rendered after the value: " m", "°", " Ω", ...


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
    # Same idea for the result-panel readouts. Fan_dipole has a per-band
    # repeat group in its result block, so it stays on hand-rolled JSX.
    # Kept independent of `legacy_controls` because in general the input
    # and output panels are unrelated concerns.
    legacy_results: bool = False
    # Mixed sequence of ParamSpec (scalar) and ParamGroupSpec (repeat).
    # The Any erases the union; runtime discrimination is by presence of
    # `params` (groups have it, scalars don't).
    param_schema: tuple[Any, ...] = field(default_factory=tuple)
    result_schema: tuple[ResultFieldSpec, ...] = field(default_factory=tuple)
    # Design-frequency band tabs offered by the UI. Defaults to the HF
    # amateur set; multi-band examples (fan_dipole) set this to () to
    # suppress the row.
    bands: tuple[BandSpec, ...] = DEFAULT_HF_BANDS
    # Optional override for the measurement-freq slider span. When None,
    # the UI uses a generic ±20%/+25% window around the design freq.
    # Multi-band examples that span the whole HF range set this to e.g.
    # (13.5, 30.2) so the slider can reach every band.
    meas_freq_range_mhz: Optional[tuple[float, float]] = None
    # Initial 2D-view projection the wire-render canvas picks when the
    # user first selects this example. "xy" = top-down (beam-in-xy
    # antennas like yagi/moxon/hexbeam); "yz" = side (antennas whose
    # arms run along y and droop in z, like inverted_v / fan_dipole;
    # also the vertical-loop hentenna). The user can still override
    # via the projection buttons; this just sets the starting view.
    default_view: str = "xy"
