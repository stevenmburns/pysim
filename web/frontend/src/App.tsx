import { useEffect, useMemo, useRef, useState } from "react";

type Wire = {
  label: string;
  knot_positions: [number, number, number][];
  knot_currents_re: number[];
  knot_currents_im: number[];
};

type Geometry = "inverted_v" | "yagi" | "moxon" | "hexbeam" | "fan_dipole" | "hentenna";

const GEOMETRY_OPTIONS: { id: Geometry; label: string }[] = [
  { id: "inverted_v", label: "Inverted V" },
  { id: "yagi", label: "Yagi" },
  { id: "moxon", label: "Moxon" },
  { id: "hexbeam", label: "Hexbeam" },
  { id: "fan_dipole", label: "Fan dipole" },
  { id: "hentenna", label: "Hentenna" },
];

type SolveResponse = {
  geometry: Geometry;
  wires: Wire[];
  feed_wire_index: number;
  feed_knot_index: number;
  z_in_re: number;
  z_in_im: number;
  design_freq_mhz: number;
  measurement_freq_mhz: number;
  lambda_design_m: number;
  solve_ms: number;
  directivity_norm?: number;
  ground?: boolean;
  height_m?: number;
  ground_eps_r?: number;
  ground_sigma?: number;
  // V-specific
  arm_len_m?: number;
  // Yagi-specific
  driver_length_m?: number;
  reflector_length_m?: number;
  spacing_m?: number;
  // Moxon-specific
  long_m?: number;
  short_m?: number;
  tipspacer_m?: number;
  t0_m?: number;
  halfdriver_m?: number;
  // Hexbeam-specific
  radius_m?: number;
  t1_m?: number;
  // Fan dipole-specific
  n_bands?: number;
  band_lengths_m?: number[];
  band_freqs_mhz?: number[];
  slope?: number;
  cone_radius_m?: number;
  // Hentenna-specific
  half_width_m?: number;
  top_height_m?: number;
  mid_offset_m?: number;
};

// Backend selector — three PySim model variants + PyNEC. Per-backend
// `model_options` are forwarded to server.py's _make_pysim_sim.
type Backend = "triangular" | "sinusoidal" | "bspline" | "pynec";

const BACKEND_LABEL: Record<Backend, string> = {
  triangular: "Triangular",
  sinusoidal: "Sinusoidal",
  bspline: "B-spline",
  pynec: "PyNEC",
};

const BACKEND_ORDER: Backend[] = ["triangular", "sinusoidal", "bspline", "pynec"];

// All three pysim models have the PEC image-method ground; PyNEC uses
// its own Sommerfeld / reflection-coefficient ground. Kept as an explicit
// list so future backends without ground support can be excluded by name.
function backendSupportsGround(b: Backend): boolean {
  return (
    b === "triangular" || b === "bspline" || b === "sinusoidal" || b === "pynec"
  );
}

type CommonOpts = { nPerWire: number; wireRadius: number };

type TriangularOpts = CommonOpts & { nQpReg: number; nQpOff: number };
type SinusoidalOpts = CommonOpts & { nQpConst: number };
type BSplineOpts = CommonOpts & {
  degree: 1 | 2;
  nQpPair: number;
  feedSmoothingFactor: number | null; // null = sharp delta-gap
  useSingularEnrichment: boolean;
  nQpSing: number;
  enrichmentMinK: number;
  nQpSource: number;
};
type PyNECOpts = CommonOpts;

type BackendOptsMap = {
  triangular: TriangularOpts;
  sinusoidal: SinusoidalOpts;
  bspline: BSplineOpts;
  pynec: PyNECOpts;
};

const DEFAULT_BACKEND_OPTS: BackendOptsMap = {
  triangular: { nPerWire: 30, wireRadius: 0.0005, nQpReg: 4, nQpOff: 4 },
  sinusoidal: { nPerWire: 30, wireRadius: 0.0005, nQpConst: 8 },
  bspline: {
    nPerWire: 30,
    wireRadius: 0.0005,
    degree: 2,
    nQpPair: 4,
    feedSmoothingFactor: null,
    useSingularEnrichment: false,
    nQpSing: 32,
    enrichmentMinK: 3,
    nQpSource: 16,
  },
  pynec: { nPerWire: 30, wireRadius: 0.0005 },
};

// Three abstract solver slots. Each holds one backend choice and its
// options; the user picks A/B/C with the row of buttons, configures the
// inhabitants from the per-slot gear menu. Lets the same UI compare
// e.g. "Triangular @ N=40" against "B-spline @ N=21 with enrichment"
// without losing either setup.
type Slot = "A" | "B" | "C";
const SLOT_ORDER: Slot[] = ["A", "B", "C"];

type SlotConfig = {
  backend: Backend;
  opts: BackendOptsMap[Backend];
};

const DEFAULT_SLOTS: Record<Slot, SlotConfig> = {
  A: {
    backend: "triangular",
    opts: { ...DEFAULT_BACKEND_OPTS.triangular, nPerWire: 40 },
  },
  B: {
    backend: "bspline",
    opts: {
      ...DEFAULT_BACKEND_OPTS.bspline,
      nPerWire: 21,
      useSingularEnrichment: true,
    },
  },
  C: {
    backend: "pynec",
    opts: { ...DEFAULT_BACKEND_OPTS.pynec, nPerWire: 41 },
  },
};

// Translates the camelCase frontend options into the snake_case kwargs the
// server forwards to each PySim model class constructor.
function modelOptionsForRequest(
  backend: Backend,
  opts: BackendOptsMap[Backend],
): Record<string, unknown> {
  if (backend === "triangular") {
    const o = opts as TriangularOpts;
    return { n_qp_reg: o.nQpReg, n_qp_off: o.nQpOff };
  }
  if (backend === "sinusoidal") {
    const o = opts as SinusoidalOpts;
    return { n_qp_const: o.nQpConst };
  }
  if (backend === "bspline") {
    const o = opts as BSplineOpts;
    return {
      degree: o.degree,
      n_qp_pair: o.nQpPair,
      n_qp_source: o.nQpSource,
      feed_smoothing_factor: o.feedSmoothingFactor,
      use_singular_enrichment: o.useSingularEnrichment,
      n_qp_sing: o.nQpSing,
      enrichment_min_k: o.enrichmentMinK,
    };
  }
  return {};
}

type SolveRequest = {
  geometry: Geometry;
  solver: "pysim" | "pynec";
  pysim_model?: "triangular" | "sinusoidal" | "bspline";
  model_options?: Record<string, unknown>;
  n_per_wire: number;
  design_freq_mhz: number;
  measurement_freq_mhz: number;
  wire_radius: number;
  ground: boolean;
  ground_fast: boolean;
  height_m: number;
  // V
  angle_deg?: number;
  halfdriver_factor?: number;
  // Yagi
  driver_length_factor?: number;
  reflector_length_factor?: number;
  spacing_wavelengths?: number;
  n_directors?: number;
  director_spacing_wavelengths?: number;
  director_size_factor?: number;
  // Moxon (+ hexbeam: hexbeam reuses tipspacer_factor and t0_factor too)
  aspect_ratio?: number;
  tipspacer_factor?: number;
  t0_factor?: number;
  // Fan dipole
  n_bands?: number;
  band_lengths_m?: number[];
  band_freqs_mhz?: number[];
  band_halfdriver_factors?: number[];
  slope?: number;
  cone_radius_m?: number;
  // Hentenna
  width_factor?: number;
  top_height_factor?: number;
  mid_height_factor?: number;
};

// Amateur HF bands the user can design for. Slider min/max snap to the
// selected band's edges; the default is the band centre. Geometry choice
// is independent of band — shape factors are dimensionless and scale to
// whatever wavelength the design freq picks.
type Band = "20m" | "17m" | "15m" | "12m" | "10m";
const BANDS: { id: Band; min: number; max: number; default: number }[] = [
  { id: "20m", min: 14.000, max: 14.350, default: 14.300 },
  { id: "17m", min: 18.068, max: 18.168, default: 18.1575 },
  { id: "15m", min: 21.000, max: 21.450, default: 21.383 },
  { id: "12m", min: 24.890, max: 24.990, default: 24.970 },
  { id: "10m", min: 28.000, max: 29.700, default: 28.470 },
];
const BAND_BY_ID: Record<Band, (typeof BANDS)[number]> = Object.fromEntries(
  BANDS.map((b) => [b.id, b]),
) as Record<Band, (typeof BANDS)[number]>;

// For a given freq, snap to the ham band whose default is closest. If the
// freq is well outside every band (e.g. user dragged a slider into a gap),
// returns the band whose center is nearest.
function nearestBandForFreq(freq: number): Band {
  let best: Band = BANDS[0].id;
  let bestDist = Infinity;
  for (const b of BANDS) {
    const d = Math.abs(freq - b.default);
    if (d < bestDist) {
      bestDist = d;
      best = b.id;
    }
  }
  return best;
}

const C_LIGHT = 299_792_458.0;
// Half-wave dipole tip-to-tip length in metres for a target freq (MHz) and
// half-arm length factor (typically ~0.95-0.97). Matches the
// `halfdriver_factor` convention used elsewhere in the codebase:
// halfdriver = factor * lambda/4, so band_length_m = 2*halfdriver.
function bandLengthFromFreqFactor(freqMhz: number, halfdriverFactor: number): number {
  const lambda = C_LIGHT / (freqMhz * 1e6);
  return (halfdriverFactor * lambda) / 2;
}

type SweepData = {
  freqs_mhz: number[];
  z_re: number[];
  z_im: number[];
};

type PatternData = {
  theta_deg: number[];
  phi_deg: number[];
  gain_dbi: number[][];
  measurement_freq_mhz: number;
};

const WS_URL = `ws://${window.location.host}/ws`;

type View = "antenna" | "azimuth" | "elevation" | "smith";
const VIEWS: { id: View; label: string }[] = [
  { id: "antenna", label: "Antenna" },
  { id: "azimuth", label: "Azimuth (xy)" },
  { id: "elevation", label: "Elevation (yz)" },
  { id: "smith", label: "Smith" },
];

// Antenna-canvas camera projections. Pick two world axes to map to canvas
// (horizontal, vertical) and project. The hidden axis is the camera ray.
type Projection = "xy" | "xz" | "yz";
const PROJECTIONS: { id: Projection; label: string; horizAxis: 0|1|2; vertAxis: 0|1|2 }[] = [
  { id: "xy", label: "Top (xy)",   horizAxis: 0, vertAxis: 1 },
  { id: "xz", label: "Front (xz)", horizAxis: 0, vertAxis: 2 },
  { id: "yz", label: "Side (yz)",  horizAxis: 1, vertAxis: 2 },
];

function defaultProjection(geometry: Geometry): Projection {
  // V-and-fan-dipole arms run along y and droop in z, so a side (yz) view is
  // the natural one. Yagi / moxon / hexbeam are top-down (xy) because the
  // beam axis lives in the xy plane. Hentenna lives entirely in the yz
  // plane (vertical rectangular loop), so yz is the only useful view.
  if (
    geometry === "inverted_v" ||
    geometry === "fan_dipole" ||
    geometry === "hentenna"
  )
    return "yz";
  return "xy";
}

function useSlideSize(maxSize = 720) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState(maxSize);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      const s = Math.min(rect.width, rect.height, maxSize);
      setSize(Math.max(160, Math.floor(s) - 16));
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [maxSize]);
  return { ref, size };
}

function useThumbColumnSize(
  stripRef: React.RefObject<HTMLDivElement>,
  maxThumb = 280,
) {
  // Vertical thumbstrip: each of 3 thumbs takes ~1/3 of the strip's actual
  // rendered height. Fixed overhead per fit:
  //   strip padding (10+10) + gaps between thumbs (2*8) +
  //   per-thumb (button padding 10 + label ~14 + gap 4 + border 2) * 3 ≈ 126
  const [size, setSize] = useState(180);
  useEffect(() => {
    const el = stripRef.current;
    if (!el) return;
    const update = () => {
      const h = el.clientHeight;
      if (h <= 0) return;
      const perThumb = (h - 130) / 3;
      setSize(Math.max(100, Math.min(maxThumb, Math.floor(perThumb))));
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [stripRef, maxThumb]);
  return size;
}

// Default fan dipole presets: 5 amateur bands ordered high-band → low-band,
// so n_bands=2 gives a maximally-distinct visual (20m + 10m) without the
// user touching any per-band sliders. Lengths from antenna_designer's
// canonical 5-band cone design.
const FAN_BAND_IDS_DEFAULT: Band[] = ["20m", "10m", "17m", "12m", "15m"];
const FAN_BAND_FREQS_DEFAULT = FAN_BAND_IDS_DEFAULT.map((id) => BAND_BY_ID[id].default);
const FAN_HALFDRIVER_FACTOR_DEFAULT = 0.962;

export function App() {
  const [geometry, setGeometry] = useState<Geometry>("inverted_v");
  // V controls
  const [angle, setAngle] = useState(30);
  const [halfdriverFactor, setHalfdriverFactor] = useState(0.962);
  // Yagi controls
  const [driverLengthFactor, setDriverLengthFactor] = useState(0.962);
  const [reflectorLengthFactor, setReflectorLengthFactor] = useState(1.01);
  const [spacingWavelengths, setSpacingWavelengths] = useState(0.15);
  const [nDirectors, setNDirectors] = useState(0);
  const [directorSpacingWavelengths, setDirectorSpacingWavelengths] = useState(0.2);
  const [directorSizeFactor, setDirectorSizeFactor] = useState(0.95);
  // Moxon controls (matching antenna_designer's canonical 28.57 MHz design).
  const [moxonHalfdriverFactor, setMoxonHalfdriverFactor] = useState(0.962);
  const [moxonAspectRatio, setMoxonAspectRatio] = useState(0.3646);
  const [moxonTipspacerFactor, setMoxonTipspacerFactor] = useState(0.0773);
  const [moxonT0Factor, setMoxonT0Factor] = useState(0.4078);
  // Hexbeam controls (matching antenna_designer's canonical 28.47 MHz design).
  // halfdriver_factor is >1 here because the hexagonal driver path is longer
  // than a straight λ/4 driver of the same resonance.
  const [hexbeamHalfdriverFactor, setHexbeamHalfdriverFactor] = useState(1.071);
  const [hexbeamTipspacerFactor, setHexbeamTipspacerFactor] = useState(0.1312);
  const [hexbeamT0Factor, setHexbeamT0Factor] = useState(0.1243);
  // Hentenna controls (antenna_designer params_50, tuned for ~50 Ω at 28.47 MHz).
  // width_factor: cross-bar half-length / lambda.
  // top_height_factor: vertical rectangle height (top to bottom edge) / lambda.
  // mid_height_factor: vertical position of cross-bar above bottom / lambda.
  const [hentennaWidthFactor, setHentennaWidthFactor] = useState(0.1378);
  const [hentennaTopHeightFactor, setHentennaTopHeightFactor] = useState(0.5081);
  const [hentennaMidHeightFactor, setHentennaMidHeightFactor] = useState(0.1094);
  // Fan dipole. Per-band state is sized at 5 so changing nBands preserves
  // inactive sliders' values when the user dials it back up.
  const [fanNBands, setFanNBands] = useState(2);
  const [fanBandIds, setFanBandIds] = useState<Band[]>([...FAN_BAND_IDS_DEFAULT]);
  const [fanBandFreqs, setFanBandFreqs] = useState<number[]>([...FAN_BAND_FREQS_DEFAULT]);
  const [fanHalfdriverFactors, setFanHalfdriverFactors] = useState<number[]>(
    Array(5).fill(FAN_HALFDRIVER_FACTOR_DEFAULT)
  );
  const [fanSlope, setFanSlope] = useState(0.5);
  const [fanConeRadius, setFanConeRadius] = useState(0.12);

  // Derived: band lengths from per-band freq * halfdriver_factor convention.
  // halfdriver = factor * lambda/4 → band_length_m = factor * lambda/2.
  // Memoized so the array reference is stable while inputs are — otherwise
  // the three useEffects below that depend on fanBandLengths would re-fire
  // on every render, flooding the WS with solve/sweep requests.
  const fanBandLengths = useMemo(
    () =>
      fanBandFreqs.map((f, i) =>
        bandLengthFromFreqFactor(f, fanHalfdriverFactors[i])
      ),
    [fanBandFreqs, fanHalfdriverFactors]
  );
  // Solver slots A / B / C — each one holds its own backend + options so
  // the user can switch between configured solvers with a single click
  // and tune each one independently from its gear menu.
  const [activeSlot, setActiveSlot] = useState<Slot>("A");
  const [slots, setSlots] = useState<Record<Slot, SlotConfig>>(DEFAULT_SLOTS);
  const [gearOpen, setGearOpen] = useState<Slot | null>(null);
  const activeConfig = slots[activeSlot];
  const backend = activeConfig.backend;
  const currentOpts = activeConfig.opts;
  const nPerWire = currentOpts.nPerWire;
  const wireRadius = currentOpts.wireRadius;
  // Stable hash of the active slot's config so useEffect can depend on it.
  const backendOptsKey = JSON.stringify(activeConfig);
  function updateSlotOpts(slot: Slot, patch: Partial<BackendOptsMap[Backend]>) {
    setSlots((prev) => ({
      ...prev,
      [slot]: {
        ...prev[slot],
        opts: { ...prev[slot].opts, ...patch } as BackendOptsMap[Backend],
      },
    }));
  }
  function setSlotBackend(slot: Slot, newBackend: Backend) {
    // Preserve segments-per-wire and wire-radius across the swap so the
    // user keeps their geometry-sizing choices when comparing models;
    // model-specific kwargs revert to that backend's defaults.
    setSlots((prev) => {
      const prevOpts = prev[slot].opts;
      const defaults = DEFAULT_BACKEND_OPTS[newBackend];
      return {
        ...prev,
        [slot]: {
          backend: newBackend,
          opts: {
            ...defaults,
            nPerWire: prevOpts.nPerWire,
            wireRadius: prevOpts.wireRadius,
          } as BackendOptsMap[Backend],
        },
      };
    });
  }
  function resetSlot(slot: Slot) {
    setSlots((prev) => ({ ...prev, [slot]: DEFAULT_SLOTS[slot] }));
  }
  const [band, setBand] = useState<Band>("20m");
  const [designFreq, setDesignFreq] = useState(BAND_BY_ID["20m"].default);
  const [measFreq, setMeasFreq] = useState(BAND_BY_ID["20m"].default);
  const [linkMeas, setLinkMeas] = useState(true);
  // Ground plane (PyNEC only). Geometry is lifted by heightM when enabled.
  const [groundEnabled, setGroundEnabled] = useState(false);
  const [groundFast, setGroundFast] = useState(false);
  const [heightM, setHeightM] = useState(7.0);
  // Far-field cut angles. The azimuth plot slices the pattern at elevation
  // `azElevDeg`; the elevation plot slices the vertical plane at azimuth
  // bearing `elevAzDeg` (0° = +x). Defaults give the conventional views.
  const [azElevDeg, setAzElevDeg] = useState(15);
  // Default elevation-cut azimuth is 0° (+x) for every geometry: Yagi,
  // moxon, and hexbeam beam +x; the inverted V now runs its arms along
  // ±y so its broadside lobe also peaks at ±x.
  const [elevAzDeg, setElevAzDeg] = useState(0);

  // When linked, design and measurement freq move together.
  function updateDesignFreq(v: number) {
    setDesignFreq(v);
    if (linkMeas) setMeasFreq(v);
  }
  function toggleLink(next: boolean) {
    setLinkMeas(next);
    if (next) setMeasFreq(designFreq);
  }

  // Pick a ham band for fan-dipole slot `i`: snaps that slot's design freq
  // to the band's default. Length-factor is preserved; user can still drag
  // the freq slider for fine tuning. When linkMeas is on, the measurement
  // freq follows the band being adjusted so the user sees the simulation
  // at the band they're currently tuning.
  function setFanBandSlot(i: number, bandId: Band) {
    setFanBandIds((prev) => {
      const next = prev.slice();
      next[i] = bandId;
      return next;
    });
    const newFreq = BAND_BY_ID[bandId].default;
    setFanBandFreqs((prev) => {
      const next = prev.slice();
      next[i] = newFreq;
      return next;
    });
    if (linkMeas) setMeasFreq(newFreq);
  }

  // Freq slider for slot `i`. Also re-snaps the pulldown to whichever band
  // is closest, so the dropdown's selected option tracks the slider value
  // even when the user drags freely between bands.
  function setFanBandFreq(i: number, v: number) {
    setFanBandFreqs((prev) => {
      const next = prev.slice();
      next[i] = v;
      return next;
    });
    setFanBandIds((prev) => {
      const nearest = nearestBandForFreq(v);
      if (prev[i] === nearest) return prev;
      const next = prev.slice();
      next[i] = nearest;
      return next;
    });
    if (linkMeas) setMeasFreq(v);
  }

  function setFanHalfdriverFactor(i: number, v: number) {
    setFanHalfdriverFactors((prev) => {
      const next = prev.slice();
      next[i] = v;
      return next;
    });
    // Tuning a band's length factor → user is focused on that band; jump
    // measurement freq to that band's current design freq so the live
    // simulation tracks the band being tuned.
    if (linkMeas) setMeasFreq(fanBandFreqs[i]);
  }

  const [result, setResult] = useState<SolveResponse | null>(null);
  const [status, setStatus] = useState<"connecting" | "open" | "closed">("connecting");
  const [rttMs, setRttMs] = useState<number | null>(null);
  const [sweep, setSweep] = useState<SweepData | null>(null);
  const [sweepRunning, setSweepRunning] = useState(false);
  // NEC's rp_card pattern, fetched on a debounce so we don't fire one per
  // slider tick. Overlaid on the cuts as a comparison line.
  const [pattern, setPattern] = useState<PatternData | null>(null);
  const [view, setView] = useState<View>("antenna");
  const [cameraProjection, setCameraProjection] = useState<Projection>(() =>
    defaultProjection("inverted_v")
  );
  // When the user switches antennas, reset the camera to that geometry's
  // natural view (V/fan_dipole → side; Yagi/moxon/hexbeam → top). Explicit
  // user override sticks until the next geometry change.
  useEffect(() => {
    setCameraProjection(defaultProjection(geometry));
  }, [geometry]);

  // For fan_dipole the antenna doesn't have a single "design frequency" —
  // each band has its own. The global designFreq state is still used by the
  // canvas (λ/4 reference bar) and by the request builder, so we drive it
  // from band 0's freq while fan_dipole is selected so the ref bar tracks
  // the longest/lowest band as the user adjusts it.
  useEffect(() => {
    if (geometry === "fan_dipole") {
      const f0 = fanBandFreqs[0];
      setDesignFreq(f0);
      if (linkMeas) setMeasFreq(f0);
    }
  }, [geometry, fanBandFreqs[0], linkMeas]);
  // Antenna-canvas current visualization is split into two independent
  // toggles: the per-segment current-magnitude heatmap (wire color/width)
  // and the |I| envelope curve overlay. Either or both can be turned off;
  // the wires and feed marker are always drawn.
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showEnvelope, setShowEnvelope] = useState(true);
  const { ref: slideRef, size: chartSize } = useSlideSize(720);
  const thumbStripRef = useRef<HTMLDivElement>(null);
  const thumbSize = useThumbColumnSize(thumbStripRef, 280);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      const idx = VIEWS.findIndex((v) => v.id === view);
      const next = e.key === "ArrowDown" ? (idx + 1) % VIEWS.length : (idx - 1 + VIEWS.length) % VIEWS.length;
      setView(VIEWS[next].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view]);

  const sweepTimerRef = useRef<number | null>(null);
  const sweepAbortRef = useRef<AbortController | null>(null);
  const patternTimerRef = useRef<number | null>(null);
  const patternAbortRef = useRef<AbortController | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const inFlightRef = useRef(false);
  const pendingRef = useRef<SolveRequest | null>(null);
  const sendStartRef = useRef(0);

  function buildRequest(): SolveRequest {
    // Solver-family ground notes: PyNEC uses Sommerfeld-Norton (or the
    // fast reflection-coefficient approximation) with εr=10, σ=0.002;
    // pysim's Triangular and B-spline models use the PEC image method.
    // Sinusoidal is free-space-only — the gear UI grays the toggle for it.
    const groundActive = groundEnabled && backendSupportsGround(backend);
    const base: SolveRequest = {
      geometry,
      solver: backend === "pynec" ? "pynec" : "pysim",
      n_per_wire: nPerWire,
      design_freq_mhz: designFreq,
      measurement_freq_mhz: measFreq,
      wire_radius: wireRadius,
      ground: groundActive,
      ground_fast: groundActive && groundFast,
      height_m: heightM,
    };
    if (backend !== "pynec") {
      base.pysim_model = backend;
      const opts = modelOptionsForRequest(backend, currentOpts);
      // BSplinePySim rejects ground_z + use_singular_enrichment together
      // (image reaction for enrichment bases isn't worked out yet). Force
      // enrichment off in the request when ground is active so the user
      // gets a sensible solve instead of a server error; the gear shows
      // an inline note.
      if (backend === "bspline" && groundActive) {
        opts.use_singular_enrichment = false;
      }
      base.model_options = opts;
    }
    if (geometry === "inverted_v") {
      base.angle_deg = angle;
      base.halfdriver_factor = halfdriverFactor;
    } else if (geometry === "yagi") {
      base.driver_length_factor = driverLengthFactor;
      base.reflector_length_factor = reflectorLengthFactor;
      base.spacing_wavelengths = spacingWavelengths;
      base.n_directors = nDirectors;
      base.director_spacing_wavelengths = directorSpacingWavelengths;
      base.director_size_factor = directorSizeFactor;
    } else if (geometry === "moxon") {
      base.halfdriver_factor = moxonHalfdriverFactor;
      base.aspect_ratio = moxonAspectRatio;
      base.tipspacer_factor = moxonTipspacerFactor;
      base.t0_factor = moxonT0Factor;
    } else if (geometry === "fan_dipole") {
      base.n_bands = fanNBands;
      base.band_lengths_m = fanBandLengths.slice(0, fanNBands);
      base.band_freqs_mhz = fanBandFreqs.slice(0, fanNBands);
      base.band_halfdriver_factors = fanHalfdriverFactors.slice(0, fanNBands);
      base.slope = fanSlope;
      base.cone_radius_m = fanConeRadius;
    } else if (geometry === "hentenna") {
      base.width_factor = hentennaWidthFactor;
      base.top_height_factor = hentennaTopHeightFactor;
      base.mid_height_factor = hentennaMidHeightFactor;
    } else {
      base.halfdriver_factor = hexbeamHalfdriverFactor;
      base.tipspacer_factor = hexbeamTipspacerFactor;
      base.t0_factor = hexbeamT0Factor;
    }
    return base;
  }

  function selectBand(next: Band) {
    setBand(next);
    const d = BAND_BY_ID[next].default;
    setDesignFreq(d);
    if (linkMeas) setMeasFreq(d);
    else if (measFreq < BAND_BY_ID[next].min || measFreq > BAND_BY_ID[next].max) {
      setMeasFreq(d);
    }
  }

  // Measurement-band quick selector: jumps measFreq to the band centre and
  // auto-unlinks from design so the antenna geometry isn't retuned.
  function selectMeasBand(next: Band) {
    if (linkMeas) setLinkMeas(false);
    setMeasFreq(BAND_BY_ID[next].default);
  }

  // Which band (if any) currently contains the measurement freq — drives
  // the active-tab highlight on the meas-band selector. Falls outside any
  // band → no tab highlighted.
  function bandContaining(f: number): Band | null {
    for (const b of BANDS) {
      if (f >= b.min && f <= b.max) return b.id;
    }
    return null;
  }

  // The latest control values, used to send a new request when the prior one
  // completes (drops intermediate values rather than queuing them all up).
  const controlsRef = useRef<SolveRequest>(buildRequest());

  useEffect(() => {
    controlsRef.current = buildRequest();
    requestSolve();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    geometry, backend, backendOptsKey,
    angle, halfdriverFactor,
    driverLengthFactor, reflectorLengthFactor, spacingWavelengths,
    nDirectors, directorSpacingWavelengths, directorSizeFactor,
    moxonHalfdriverFactor, moxonAspectRatio, moxonTipspacerFactor, moxonT0Factor,
    hexbeamHalfdriverFactor, hexbeamTipspacerFactor, hexbeamT0Factor,
    hentennaWidthFactor, hentennaTopHeightFactor, hentennaMidHeightFactor,
    fanNBands, fanBandLengths, fanSlope, fanConeRadius,
    designFreq, measFreq,
    groundEnabled, groundFast, heightM,
  ]);

  // Debounced sweep across measurement freq. Re-runs whenever any antenna
  // parameter changes. Single-band geometries sweep around designFreq, so
  // moving the measFreq slider doesn't re-sweep (the existing data already
  // covers the new slider position). Fan dipole sweeps around measFreq,
  // so measFreq is part of the deps there to re-anchor.
  useEffect(() => {
    // Cancel any in-flight sweep fetch immediately. Without this the
    // previous sweep keeps streaming for hundreds of ms (PyNEC ground at
    // 100 ms/point × 41 points = ~4 s) and starves the live /ws solve of
    // CPU — the user moves a slider but the next impedance update is
    // delayed behind the now-stale sweep finishing.
    sweepAbortRef.current?.abort();
    if (sweepTimerRef.current) {
      window.clearTimeout(sweepTimerRef.current);
    }
    setSweep(null);
    setSweepRunning(false);
    sweepTimerRef.current = window.setTimeout(() => {
      runSweep();
      sweepTimerRef.current = null;
    }, 500);
    return () => {
      if (sweepTimerRef.current) window.clearTimeout(sweepTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    geometry, backend, backendOptsKey,
    angle, halfdriverFactor,
    driverLengthFactor, reflectorLengthFactor, spacingWavelengths,
    nDirectors, directorSpacingWavelengths, directorSizeFactor,
    moxonHalfdriverFactor, moxonAspectRatio, moxonTipspacerFactor, moxonT0Factor,
    hexbeamHalfdriverFactor, hexbeamTipspacerFactor, hexbeamT0Factor,
    hentennaWidthFactor, hentennaTopHeightFactor, hentennaMidHeightFactor,
    fanNBands, fanBandLengths, fanSlope, fanConeRadius,
    designFreq,
    groundEnabled, groundFast, heightM,
    geometry === "fan_dipole" ? measFreq : null,
  ]);

  // Debounced NEC pattern fetch. PyNEC only — for pysim there's no rp_card
  // equivalent. Tracks measurement freq too (unlike the impedance sweep).
  useEffect(() => {
    if (patternTimerRef.current) window.clearTimeout(patternTimerRef.current);
    setPattern(null);
    if (backend !== "pynec") return;
    patternTimerRef.current = window.setTimeout(() => {
      runPattern();
      patternTimerRef.current = null;
    }, 500);
    return () => {
      if (patternTimerRef.current) window.clearTimeout(patternTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    geometry, backend, backendOptsKey,
    angle, halfdriverFactor,
    driverLengthFactor, reflectorLengthFactor, spacingWavelengths,
    nDirectors, directorSpacingWavelengths, directorSizeFactor,
    moxonHalfdriverFactor, moxonAspectRatio, moxonTipspacerFactor, moxonT0Factor,
    hexbeamHalfdriverFactor, hexbeamTipspacerFactor, hexbeamT0Factor,
    hentennaWidthFactor, hentennaTopHeightFactor, hentennaMidHeightFactor,
    fanNBands, fanBandLengths, fanSlope, fanConeRadius,
    designFreq, measFreq,
    groundEnabled, groundFast, heightM,
  ]);

  async function runSweep() {
    sweepAbortRef.current?.abort();
    const controller = new AbortController();
    sweepAbortRef.current = controller;

    // Sweep range, log-spaced. Sommerfeld-Norton ground is ~100x slower
    // per point, so halve the resolution there to keep total sweep time
    // near free-space cost. Fast (reflection-coefficient) ground and pysim
    // PEC ground are cheap enough for full resolution.
    //
    // Anchor: single-band geometries sweep around designFreq; fan_dipole is
    // multi-band, so we sweep around measFreq instead — that's where the
    // user is currently probing.
    //
    // Span: multi-band antennas use ±5% (narrow, centered on the band the
    // user is tuning) — wider would cross into neighbouring band tuning
    // and clutter the Smith trajectory. Single-band antennas keep the
    // broader 0.8x..1.25x for the resonance / out-of-band picture.
    const slowGround = backend === "pynec" && groundEnabled && !groundFast;
    const N = slowGround ? 21 : 41;
    const sweepAnchor = geometry === "fan_dipole" ? measFreq : designFreq;
    const multiband = geometry === "fan_dipole";
    const loFactor = multiband ? 0.95 : 0.8;
    const hiFactor = multiband ? 1.05 : 1.25;
    const fLo = Math.max(0.5, sweepAnchor * loFactor);
    const fHi = Math.min(60, sweepAnchor * hiFactor);
    const freqs = Array.from({ length: N }, (_, i) =>
      Math.exp(Math.log(fLo) + (i / (N - 1)) * (Math.log(fHi) - Math.log(fLo))),
    );

    const body = { ...buildRequest(), freqs_mhz: freqs };
    setSweepRunning(true);
    const acc: SweepData = { freqs_mhz: [], z_re: [], z_im: [] };
    try {
      const resp = await fetch("/sweep", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) throw new Error(`sweep failed: ${resp.status}`);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const pt = JSON.parse(line);
          if (pt.done) continue;
          acc.freqs_mhz.push(pt.freq_mhz);
          acc.z_re.push(pt.z_re);
          acc.z_im.push(pt.z_im);
          if (!controller.signal.aborted) {
            // New object so React re-renders the Smith chart per point.
            setSweep({
              freqs_mhz: acc.freqs_mhz.slice(),
              z_re: acc.z_re.slice(),
              z_im: acc.z_im.slice(),
            });
          }
        }
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      console.error("sweep error", e);
    } finally {
      if (sweepAbortRef.current === controller) {
        sweepAbortRef.current = null;
        setSweepRunning(false);
      }
    }
  }

  async function runPattern() {
    patternAbortRef.current?.abort();
    const controller = new AbortController();
    patternAbortRef.current = controller;
    try {
      const resp = await fetch("/pattern", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildRequest()),
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error(`pattern failed: ${resp.status}`);
      const data = await resp.json();
      if (!data.available) {
        setPattern(null);
        return;
      }
      if (!controller.signal.aborted) setPattern(data as PatternData);
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      console.error("pattern error", e);
    } finally {
      if (patternAbortRef.current === controller) patternAbortRef.current = null;
    }
  }

  function requestSolve() {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      pendingRef.current = controlsRef.current;
      return;
    }
    if (inFlightRef.current) {
      // Coalesce: latest controls will be picked up when the response arrives.
      pendingRef.current = controlsRef.current;
      return;
    }
    inFlightRef.current = true;
    sendStartRef.current = performance.now();
    ws.send(JSON.stringify(controlsRef.current));
  }

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => {
      setStatus("open");
      // A prior socket's pending response can never arrive here; clear the
      // in-flight flag so this socket can start sending. StrictMode and HMR
      // both tear down + recreate this socket and would otherwise leave the
      // flag stuck true, blocking all subsequent slider-driven solves.
      inFlightRef.current = false;
      pendingRef.current = controlsRef.current;
      requestSolve();
    };
    ws.onclose = () => {
      setStatus("closed");
      inFlightRef.current = false;
    };
    ws.onerror = () => {
      setStatus("closed");
      inFlightRef.current = false;
    };
    ws.onmessage = (ev) => {
      setRttMs(performance.now() - sendStartRef.current);
      const data: SolveResponse = JSON.parse(ev.data);
      inFlightRef.current = false;
      setResult(data);
      // If controls changed while waiting, fire the next solve immediately.
      if (pendingRef.current) {
        pendingRef.current = null;
        requestSolve();
      }
    };
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>pysim — interactive</h1>

        <div className="geometry-select-row">
          <label className="geometry-select-label" htmlFor="geometry-select">
            antenna
          </label>
          <select
            id="geometry-select"
            className="geometry-select"
            value={geometry}
            onChange={(e) => setGeometry(e.target.value as Geometry)}
          >
            {GEOMETRY_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {geometry === "inverted_v" && (
          <>
            <div className="field">
              <label>
                <span>droop angle</span>
                <span>{angle.toFixed(1)}°</span>
              </label>
              <input
                type="range"
                min={0}
                max={80}
                step={0.5}
                value={angle}
                onInput={(e) => setAngle(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>halfdriver factor</span>
                <span>{halfdriverFactor.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.5}
                max={1.2}
                step={0.001}
                value={halfdriverFactor}
                onInput={(e) => setHalfdriverFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
          </>
        )}

        {geometry === "yagi" && (
          <>
            <div className="field">
              <label>
                <span>driver length factor</span>
                <span>{driverLengthFactor.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.5}
                max={1.2}
                step={0.001}
                value={driverLengthFactor}
                onInput={(e) => setDriverLengthFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>reflector length factor</span>
                <span>{reflectorLengthFactor.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.5}
                max={1.2}
                step={0.001}
                value={reflectorLengthFactor}
                onInput={(e) => setReflectorLengthFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>spacing (λ)</span>
                <span>{spacingWavelengths.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.001}
                value={spacingWavelengths}
                onInput={(e) => setSpacingWavelengths(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span># directors</span>
                <span>{nDirectors}</span>
              </label>
              <input
                type="range"
                min={0}
                max={8}
                step={1}
                value={nDirectors}
                onInput={(e) => setNDirectors(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            {nDirectors > 0 && (
              <>
                <div className="field">
                  <label>
                    <span>director spacing (λ)</span>
                    <span>{directorSpacingWavelengths.toFixed(3)}</span>
                  </label>
                  <input
                    type="range"
                    min={0.05}
                    max={0.5}
                    step={0.001}
                    value={directorSpacingWavelengths}
                    onInput={(e) => setDirectorSpacingWavelengths(Number((e.target as HTMLInputElement).value))}
                  />
                </div>
                <div className="field">
                  <label>
                    <span>director size factor</span>
                    <span>{directorSizeFactor.toFixed(3)}</span>
                  </label>
                  <input
                    type="range"
                    min={0.5}
                    max={1.2}
                    step={0.001}
                    value={directorSizeFactor}
                    onInput={(e) => setDirectorSizeFactor(Number((e.target as HTMLInputElement).value))}
                  />
                </div>
              </>
            )}
          </>
        )}

        {geometry === "moxon" && (
          <>
            <div className="field">
              <label>
                <span>halfdriver factor</span>
                <span>{moxonHalfdriverFactor.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.8}
                max={1.1}
                step={0.001}
                value={moxonHalfdriverFactor}
                onInput={(e) => setMoxonHalfdriverFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>aspect ratio (short/long)</span>
                <span>{moxonAspectRatio.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.2}
                max={0.6}
                step={0.001}
                value={moxonAspectRatio}
                onInput={(e) => setMoxonAspectRatio(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>tip spacer factor</span>
                <span>{moxonTipspacerFactor.toFixed(4)}</span>
              </label>
              <input
                type="range"
                min={0.02}
                max={0.20}
                step={0.0005}
                value={moxonTipspacerFactor}
                onInput={(e) => setMoxonTipspacerFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>t0 factor (tip length / short)</span>
                <span>{moxonT0Factor.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.15}
                max={0.6}
                step={0.001}
                value={moxonT0Factor}
                onInput={(e) => setMoxonT0Factor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
          </>
        )}

        {geometry === "hexbeam" && (
          <>
            <div className="field">
              <label>
                <span>halfdriver factor</span>
                <span>{hexbeamHalfdriverFactor.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.9}
                max={1.25}
                step={0.001}
                value={hexbeamHalfdriverFactor}
                onInput={(e) => setHexbeamHalfdriverFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>tip spacer factor</span>
                <span>{hexbeamTipspacerFactor.toFixed(4)}</span>
              </label>
              <input
                type="range"
                min={0.04}
                max={0.25}
                step={0.0005}
                value={hexbeamTipspacerFactor}
                onInput={(e) => setHexbeamTipspacerFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>t0 factor (tip length / radius)</span>
                <span>{hexbeamT0Factor.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.04}
                max={0.30}
                step={0.001}
                value={hexbeamT0Factor}
                onInput={(e) => setHexbeamT0Factor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
          </>
        )}

        {geometry === "hentenna" && (
          <>
            <div className="field">
              <label>
                <span>width factor</span>
                <span>{hentennaWidthFactor.toFixed(4)}</span>
              </label>
              <input
                type="range"
                min={0.05}
                max={0.30}
                step={0.0005}
                value={hentennaWidthFactor}
                onInput={(e) => setHentennaWidthFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>top height factor</span>
                <span>{hentennaTopHeightFactor.toFixed(4)}</span>
              </label>
              <input
                type="range"
                min={0.30}
                max={0.70}
                step={0.0005}
                value={hentennaTopHeightFactor}
                onInput={(e) => setHentennaTopHeightFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>mid height factor</span>
                <span>{hentennaMidHeightFactor.toFixed(4)}</span>
              </label>
              <input
                type="range"
                min={0.03}
                max={0.30}
                step={0.0005}
                value={hentennaMidHeightFactor}
                onInput={(e) => setHentennaMidHeightFactor(Number((e.target as HTMLInputElement).value))}
              />
            </div>
          </>
        )}

        {geometry === "fan_dipole" && (
          <>
            <div className="field">
              <label>
                <span># bands</span>
                <span>{fanNBands}</span>
              </label>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={fanNBands}
                onInput={(e) => setFanNBands(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            {Array.from({ length: fanNBands }, (_, i) => {
              const bandId = fanBandIds[i];
              const bandSpec = BAND_BY_ID[bandId];
              return (
                <div className="fan-band-group" key={`fan-band-${i}`}>
                  <div className="fan-band-header">
                    <span className="fan-band-label">band {i}</span>
                    <select
                      className="fan-band-select"
                      value={bandId}
                      onChange={(e) => setFanBandSlot(i, e.target.value as Band)}
                    >
                      {BANDS.map((b) => (
                        <option key={b.id} value={b.id}>{b.id}</option>
                      ))}
                    </select>
                    <span className="fan-band-readout">{fanBandFreqs[i].toFixed(3)} MHz</span>
                  </div>
                  <input
                    type="range"
                    min={bandSpec.min}
                    max={bandSpec.max}
                    step={0.001}
                    value={fanBandFreqs[i]}
                    onInput={(e) =>
                      setFanBandFreq(i, Number((e.target as HTMLInputElement).value))
                    }
                  />
                  <label className="fan-band-sublabel">
                    <span>length factor</span>
                    <span>
                      {fanHalfdriverFactors[i].toFixed(3)}{" "}
                      <span className="fan-band-len">
                        ({fanBandLengths[i].toFixed(2)} m tip-to-tip)
                      </span>
                    </span>
                  </label>
                  <input
                    type="range"
                    min={0.85}
                    max={1.05}
                    step={0.001}
                    value={fanHalfdriverFactors[i]}
                    onInput={(e) =>
                      setFanHalfdriverFactor(
                        i,
                        Number((e.target as HTMLInputElement).value)
                      )
                    }
                  />
                </div>
              );
            })}
            <div className="field">
              <label>
                <span>cone slope</span>
                <span>{fanSlope.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={0.0}
                max={1.5}
                step={0.01}
                value={fanSlope}
                onInput={(e) => setFanSlope(Number((e.target as HTMLInputElement).value))}
              />
            </div>
            <div className="field">
              <label>
                <span>cone radius</span>
                <span>{fanConeRadius.toFixed(3)} m</span>
              </label>
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.005}
                value={fanConeRadius}
                onInput={(e) => setFanConeRadius(Number((e.target as HTMLInputElement).value))}
              />
            </div>
          </>
        )}

        {geometry !== "fan_dipole" && (
          <div className="field">
            <label>
              <span>design freq</span>
              <span>{designFreq.toFixed(3)} MHz</span>
            </label>
            <div className="geometry-tabs band-tabs" role="tablist">
              {BANDS.map((b) => (
                <button
                  key={b.id}
                  role="tab"
                  aria-selected={band === b.id}
                  className={band === b.id ? "active" : ""}
                  onClick={() => selectBand(b.id)}
                >
                  {b.id}
                </button>
              ))}
            </div>
            <input
              type="range"
              min={BAND_BY_ID[band].min}
              max={BAND_BY_ID[band].max}
              step={0.005}
              value={designFreq}
              onInput={(e) => updateDesignFreq(Number((e.target as HTMLInputElement).value))}
            />
          </div>
        )}

        <div className="group-label">simulation</div>

        <div className="field">
          <label>
            <span>solver slot</span>
            <span>{BACKEND_LABEL[backend]} · N={nPerWire}</span>
          </label>
          <div className="backend-tabs" role="tablist">
            {SLOT_ORDER.map((s) => {
              const cfg = slots[s];
              return (
                <div key={s} className="backend-tab-cell">
                  <button
                    role="tab"
                    aria-selected={activeSlot === s}
                    className={`backend-tab-btn ${activeSlot === s ? "active" : ""}`}
                    title={`${BACKEND_LABEL[cfg.backend]}, N=${cfg.opts.nPerWire}`}
                    onClick={() => setActiveSlot(s)}
                  >
                    <span className="slot-letter">{s}</span>
                    <span className="slot-sub">{BACKEND_LABEL[cfg.backend]}</span>
                  </button>
                  <button
                    className="backend-gear-btn"
                    title={`Slot ${s} options`}
                    aria-label={`Slot ${s} options`}
                    onClick={() => setGearOpen(s)}
                  >
                    ⚙
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {!backendSupportsGround(backend) && groundEnabled && (
          <div className="field" title="This backend doesn't model ground; ignored until you switch to one that does.">
            <em style={{ color: "var(--muted)", fontSize: 12 }}>
              ground plane ignored for {BACKEND_LABEL[backend]}
            </em>
          </div>
        )}

        <div className="field">
          <label
            className="link-toggle"
            title={
              backend === "pynec"
                ? "Sommerfeld-Norton ground (εr=10, σ=0.002 S/m)"
                : "PEC image-method ground (perfect electric conductor)"
            }
          >
            <input
              type="checkbox"
              checked={groundEnabled}
              disabled={!backendSupportsGround(backend)}
              onChange={(e) => setGroundEnabled(e.target.checked)}
            />
            ground plane{" "}
            {backend === "pynec"
              ? "(εr=10, σ=0.002 S/m)"
              : "(PEC, perfect conductor)"}
          </label>
          {backend === "pynec" && groundEnabled && (
            <label
              className="link-toggle"
              title="Reflection-coefficient approximation (NEC ITYPE=0). ~10x faster per solve than Sommerfeld-Norton; degrades for very-low antennas near the horizon."
            >
              <input
                type="checkbox"
                checked={groundFast}
                onChange={(e) => setGroundFast(e.target.checked)}
              />
              fast ground (reflection coefficient)
            </label>
          )}
        </div>

        {groundEnabled && (
          <div className="field">
            <label>
              <span>height above ground</span>
              <span>{heightM.toFixed(2)} m</span>
            </label>
            <input
              type="range"
              min={0.5}
              max={30}
              step={0.1}
              value={heightM}
              onInput={(e) => setHeightM(Number((e.target as HTMLInputElement).value))}
            />
          </div>
        )}

        <div className="field">
          <label>
            <span>measurement freq</span>
            <span>{measFreq.toFixed(3)} MHz</span>
          </label>
          {/* Fan dipole is multi-band, so the slider has to span all five
              bands rather than ±25% of a single design freq. */}
          <div className="geometry-tabs band-tabs" role="tablist">
            {BANDS.map((b) => {
              const active = bandContaining(measFreq) === b.id;
              return (
                <button
                  key={b.id}
                  role="tab"
                  aria-selected={active}
                  className={active ? "active" : ""}
                  onClick={() => selectMeasBand(b.id)}
                >
                  {b.id}
                </button>
              );
            })}
          </div>
          <input
            type="range"
            min={
              geometry === "fan_dipole"
                ? BANDS[0].min - 0.5
                : Math.max(0.5, designFreq * 0.8)
            }
            max={
              geometry === "fan_dipole"
                ? BANDS[BANDS.length - 1].max + 0.5
                : Math.min(60, designFreq * 1.25)
            }
            step={0.005}
            value={measFreq}
            disabled={linkMeas}
            onInput={(e) => setMeasFreq(Number((e.target as HTMLInputElement).value))}
          />
          <label className="link-toggle">
            <input
              type="checkbox"
              checked={linkMeas}
              onChange={(e) => toggleLink(e.target.checked)}
            />
            lock to design freq
          </label>
        </div>

        <div className="group-label">far-field cuts</div>

        <div className="field">
          <label>
            <span>azimuth at elevation</span>
            <span>{azElevDeg.toFixed(0)}°</span>
          </label>
          <input
            type="range"
            min={0}
            max={89}
            step={1}
            value={azElevDeg}
            onInput={(e) => setAzElevDeg(Number((e.target as HTMLInputElement).value))}
          />
        </div>

        <div className="field">
          <label>
            <span>elevation at azimuth</span>
            <span>{elevAzDeg.toFixed(0)}°</span>
          </label>
          <input
            type="range"
            min={0}
            max={359}
            step={1}
            value={elevAzDeg}
            onInput={(e) => setElevAzDeg(Number((e.target as HTMLInputElement).value))}
          />
        </div>

        <div className="readout">
          <div className="row">
            <span>R</span>
            <span className="val">{result ? `${result.z_in_re.toFixed(2)} Ω` : "—"}</span>
          </div>
          <div className="row">
            <span>X</span>
            <span className={result && Math.abs(result.z_in_im) < 2 ? "val val-hot" : "val"}>
              {result ? `${result.z_in_im.toFixed(2)} Ω` : "—"}
            </span>
          </div>
          {result?.geometry === "inverted_v" && result.arm_len_m != null && (
            <div className="row">
              <span>arm length</span>
              <span className="val">{result.arm_len_m.toFixed(3)} m</span>
            </div>
          )}
          {result?.geometry === "yagi" && (
            <>
              <div className="row">
                <span>driver L</span>
                <span className="val">{result.driver_length_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>reflector L</span>
                <span className="val">{result.reflector_length_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>spacing</span>
                <span className="val">{result.spacing_m?.toFixed(3)} m</span>
              </div>
            </>
          )}
          {result?.geometry === "moxon" && (
            <>
              <div className="row">
                <span>long (vertical)</span>
                <span className="val">{result.long_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>short (gap)</span>
                <span className="val">{result.short_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>tip spacer</span>
                <span className="val">{result.tipspacer_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>tip length t0</span>
                <span className="val">{result.t0_m?.toFixed(3)} m</span>
              </div>
            </>
          )}
          {result?.geometry === "hexbeam" && (
            <>
              <div className="row">
                <span>radius</span>
                <span className="val">{result.radius_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>tip length t0</span>
                <span className="val">{result.t0_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>driver tip t1</span>
                <span className="val">{result.t1_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>tip spacer</span>
                <span className="val">{result.tipspacer_m?.toFixed(3)} m</span>
              </div>
            </>
          )}
          {result?.geometry === "hentenna" && (
            <>
              <div className="row">
                <span>half width</span>
                <span className="val">{result.half_width_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>top height</span>
                <span className="val">{result.top_height_m?.toFixed(3)} m</span>
              </div>
              <div className="row">
                <span>mid offset</span>
                <span className="val">{result.mid_offset_m?.toFixed(3)} m</span>
              </div>
            </>
          )}
          {result?.geometry === "fan_dipole" && (
            <>
              <div className="row">
                <span>bands</span>
                <span className="val">{result.n_bands}</span>
              </div>
              {result.band_lengths_m?.map((L, i) => (
                <div className="row" key={`fan-out-${i}`}>
                  <span>band {i + 1} ({result.band_freqs_mhz?.[i]?.toFixed(2)} MHz)</span>
                  <span className="val">{L.toFixed(3)} m</span>
                </div>
              ))}
              <div className="row">
                <span>cone slope</span>
                <span className="val">{result.slope?.toFixed(3)}</span>
              </div>
              <div className="row">
                <span>cone radius</span>
                <span className="val">{result.cone_radius_m?.toFixed(3)} m</span>
              </div>
            </>
          )}
          <div className="row">
            <span>|I_feed|</span>
            <span className="val">
              {result ? feedMag(result).toExponential(3) : "—"}
            </span>
          </div>
          <div className="row">
            <span>solve</span>
            <span className="val">{result ? `${result.solve_ms.toFixed(1)} ms` : "—"}</span>
          </div>
          <div className="row">
            <span>SWR (50 Ω)</span>
            <span className="val">{result ? formatSwr(result.z_in_re, result.z_in_im, 50) : "—"}</span>
          </div>
          <div className="row">
            <span>rtt</span>
            <span className="val">{rttMs != null ? `${rttMs.toFixed(1)} ms` : "—"}</span>
          </div>
        </div>

        {gearOpen && (
          <BackendConfigModal
            slot={gearOpen}
            backend={slots[gearOpen].backend}
            opts={slots[gearOpen].opts}
            onChangeBackend={(b) => setSlotBackend(gearOpen, b)}
            onPatch={(patch) => updateSlotOpts(gearOpen, patch)}
            onReset={() => resetSlot(gearOpen)}
            onClose={() => setGearOpen(null)}
          />
        )}
      </aside>

      <main className="stage">
        <div className="thumbstrip" ref={thumbStripRef}>
          {VIEWS.filter((v) => v.id !== view).map((v) => (
            <button
              key={v.id}
              className="thumb"
              onClick={() => setView(v.id)}
              title={`Switch to ${v.label}`}
            >
              <div
                className="thumb-canvas"
                style={{ width: thumbSize, height: thumbSize }}
              >
                <ViewPanel
                  view={v.id}
                  size={thumbSize}
                  fill={false}
                  result={result}
                  sweep={sweep}
                  pattern={pattern}
                  measFreqMhz={measFreq}
                  sweepRunning={sweepRunning}
                  azElevDeg={azElevDeg}
                  elevAzDeg={elevAzDeg}
                  cameraProjection={cameraProjection}
                  showHeatmap={showHeatmap}
                  showEnvelope={showEnvelope}
                />
              </div>
              <div className="thumb-label">{v.label}</div>
            </button>
          ))}
        </div>
        <div className="carousel-slide" ref={slideRef}>
          {view === "antenna" && (
            <div className="antenna-overlay">
              <div className="projection-toggle">
                {PROJECTIONS.map((p) => (
                  <button
                    key={p.id}
                    className={p.id === cameraProjection ? "active" : ""}
                    onClick={() => setCameraProjection(p.id)}
                    title={`Project onto the ${p.id} plane`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <label
                className="overlay-checkbox"
                title="Color wire segments by current magnitude; modulate wire width"
              >
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                />
                heatmapped currents
              </label>
              <label
                className="overlay-checkbox"
                title="Draw the |I| envelope curve along each wire"
              >
                <input
                  type="checkbox"
                  checked={showEnvelope}
                  onChange={(e) => setShowEnvelope(e.target.checked)}
                />
                current waveforms
              </label>
            </div>
          )}
          <ViewPanel
            view={view}
            size={chartSize}
            fill={view === "antenna"}
            result={result}
            sweep={sweep}
            pattern={pattern}
            measFreqMhz={measFreq}
            sweepRunning={sweepRunning}
            azElevDeg={azElevDeg}
            elevAzDeg={elevAzDeg}
            cameraProjection={cameraProjection}
            showHeatmap={showHeatmap}
            showEnvelope={showEnvelope}
          />
        </div>
        <div className="status">ws: {status}</div>
      </main>
    </div>
  );
}

type BackendConfigProps = {
  slot: Slot;
  backend: Backend;
  opts: BackendOptsMap[Backend];
  onChangeBackend: (b: Backend) => void;
  onPatch: (patch: Partial<BackendOptsMap[Backend]>) => void;
  onReset: () => void;
  onClose: () => void;
};

function BackendConfigModal({
  slot,
  backend,
  opts,
  onChangeBackend,
  onPatch,
  onReset,
  onClose,
}: BackendConfigProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="backend-config-overlay" onClick={onClose}>
      <div
        className="backend-config-modal"
        role="dialog"
        aria-label={`Slot ${slot} options`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="backend-config-header">
          <strong>Slot {slot} — {BACKEND_LABEL[backend]}</strong>
          <button className="backend-config-close" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="backend-config-body">
          <div className="field">
            <label>
              <span>solver</span>
              <span>{BACKEND_LABEL[backend]}</span>
            </label>
            <div className="geometry-tabs" role="tablist">
              {BACKEND_ORDER.map((b) => (
                <button
                  key={b}
                  role="tab"
                  aria-selected={backend === b}
                  className={backend === b ? "active" : ""}
                  onClick={() => onChangeBackend(b)}
                >
                  {BACKEND_LABEL[b]}
                </button>
              ))}
            </div>
          </div>

          <NumberField
            label="segments / wire (N)"
            value={opts.nPerWire}
            min={4}
            max={120}
            step={1}
            onChange={(v) => onPatch({ nPerWire: v })}
          />
          <NumberField
            label="wire radius (m)"
            value={opts.wireRadius}
            step={0.0001}
            onChange={(v) => onPatch({ wireRadius: v })}
          />

          {backend === "triangular" && (
            <>
              <NumberField
                label="n_qp_reg (same-edge GL pts)"
                value={(opts as TriangularOpts).nQpReg}
                min={2}
                max={16}
                step={1}
                onChange={(v) => onPatch({ nQpReg: v } as never)}
              />
              <NumberField
                label="n_qp_off (cross-edge GL pts)"
                value={(opts as TriangularOpts).nQpOff}
                min={2}
                max={16}
                step={1}
                onChange={(v) => onPatch({ nQpOff: v } as never)}
              />
            </>
          )}

          {backend === "sinusoidal" && (
            <NumberField
              label="n_qp_const (GL pts)"
              value={(opts as SinusoidalOpts).nQpConst}
              min={2}
              max={32}
              step={1}
              onChange={(v) => onPatch({ nQpConst: v } as never)}
            />
          )}

          {backend === "bspline" && (
            <BSplineFields
              opts={opts as BSplineOpts}
              onPatch={(p) => onPatch(p as never)}
            />
          )}

          {backend === "pynec" && (
            <em style={{ color: "var(--muted)", fontSize: 12 }}>
              PyNEC has no extra solver knobs here — ground type / fast ground
              live in the main panel.
            </em>
          )}
        </div>

        <div className="backend-config-footer">
          <button className="backend-config-reset" onClick={onReset}>
            reset to defaults
          </button>
        </div>
      </div>
    </div>
  );
}

function BSplineFields({
  opts,
  onPatch,
}: {
  opts: BSplineOpts;
  onPatch: (p: Partial<BSplineOpts>) => void;
}) {
  return (
    <>
      <div className="field">
        <label>
          <span>degree</span>
          <span>{opts.degree}</span>
        </label>
        <div className="geometry-tabs" role="tablist">
          {[1, 2].map((d) => (
            <button
              key={d}
              role="tab"
              aria-selected={opts.degree === d}
              className={opts.degree === d ? "active" : ""}
              onClick={() => onPatch({ degree: d as 1 | 2 })}
            >
              d={d}
            </button>
          ))}
        </div>
      </div>
      <NumberField
        label="n_qp_pair (GL pts/axis)"
        value={opts.nQpPair}
        min={2}
        max={16}
        step={1}
        onChange={(v) => onPatch({ nQpPair: v })}
      />
      <div className="field">
        <label className="link-toggle" title="Replace delta-gap source with cos² bump of width α·h_feed; basis-limited convergence on dipoles.">
          <input
            type="checkbox"
            checked={opts.feedSmoothingFactor != null}
            onChange={(e) =>
              onPatch({ feedSmoothingFactor: e.target.checked ? 3 : null })
            }
          />
          feed source smoothing
        </label>
        {opts.feedSmoothingFactor != null && (
          <NumberField
            label="α (bump width / h_feed)"
            value={opts.feedSmoothingFactor}
            min={0.5}
            max={10}
            step={0.5}
            onChange={(v) => onPatch({ feedSmoothingFactor: v })}
          />
        )}
        {opts.feedSmoothingFactor != null && (
          <NumberField
            label="n_qp_source"
            value={opts.nQpSource}
            min={4}
            max={64}
            step={1}
            onChange={(v) => onPatch({ nQpSource: v })}
          />
        )}
      </div>
      <div className="field">
        <label className="link-toggle" title="Add (u/h)·log(u/h) singular basis at K ≥ enrichment_min_k junctions; flips hentenna O(1/N) → ~O(1/N^(d+1)).">
          <input
            type="checkbox"
            checked={opts.useSingularEnrichment}
            onChange={(e) => onPatch({ useSingularEnrichment: e.target.checked })}
          />
          K≥3 junction singular enrichment
        </label>
        {opts.useSingularEnrichment && (
          <>
            <NumberField
              label="n_qp_sing (GL pts/axis)"
              value={opts.nQpSing}
              min={8}
              max={64}
              step={1}
              onChange={(v) => onPatch({ nQpSing: v })}
            />
            <NumberField
              label="enrichment_min_k"
              value={opts.enrichmentMinK}
              min={2}
              max={6}
              step={1}
              onChange={(v) => onPatch({ enrichmentMinK: v })}
            />
          </>
        )}
      </div>
    </>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="field">
      <label>
        <span>{label}</span>
        <span>{value}</span>
      </label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (!Number.isNaN(v)) onChange(v);
        }}
      />
    </div>
  );
}

function feedMag(r: SolveResponse): number {
  const w = r.wires[r.feed_wire_index];
  if (!w) return 0;
  const re = w.knot_currents_re[r.feed_knot_index];
  const im = w.knot_currents_im[r.feed_knot_index];
  return Math.hypot(re, im);
}

function reflectionCoefficient(r: number, x: number, z0: number) {
  // Γ = (Z - Z0) / (Z + Z0), with Z = r + jx (Z0 real).
  const denom = (r + z0) * (r + z0) + x * x;
  const gRe = (r * r - z0 * z0 + x * x) / denom;
  const gIm = (2 * x * z0) / denom;
  return { gRe, gIm, gMag: Math.hypot(gRe, gIm) };
}

function formatSwr(r: number, x: number, z0: number): string {
  const { gMag } = reflectionCoefficient(r, x, z0);
  if (gMag >= 0.9999) return "∞";
  const swr = (1 + gMag) / (1 - gMag);
  if (swr > 99) return swr.toFixed(0);
  return swr.toFixed(2);
}

function ViewPanel({
  view,
  size,
  fill,
  result,
  sweep,
  pattern,
  measFreqMhz,
  sweepRunning,
  azElevDeg,
  elevAzDeg,
  cameraProjection,
  showHeatmap,
  showEnvelope,
}: {
  view: View;
  size: number;
  fill: boolean;
  result: SolveResponse | null;
  sweep: SweepData | null;
  pattern: PatternData | null;
  measFreqMhz: number;
  sweepRunning: boolean;
  azElevDeg: number;
  elevAzDeg: number;
  cameraProjection: Projection;
  showHeatmap: boolean;
  showEnvelope: boolean;
}) {
  if (view === "antenna") {
    return (
      <div className={fill ? "antenna-fill" : "antenna-thumb"}
           style={fill ? undefined : { width: size, height: size }}>
        <CurrentCanvas
          result={result}
          projection={cameraProjection}
          showHeatmap={showHeatmap}
          showEnvelope={showEnvelope}
        />
      </div>
    );
  }
  if (view === "azimuth") {
    return (
      <FarFieldChart
        result={result}
        pattern={pattern}
        size={size}
        cut="xy"
        azElevDeg={azElevDeg}
        elevAzDeg={elevAzDeg}
      />
    );
  }
  if (view === "elevation") {
    return (
      <FarFieldChart
        result={result}
        pattern={pattern}
        size={size}
        cut="yz"
        azElevDeg={azElevDeg}
        elevAzDeg={elevAzDeg}
      />
    );
  }
  return (
    <SmithChart
      r={result?.z_in_re ?? 0}
      x={result?.z_in_im ?? 0}
      z0={50}
      size={size}
      sweep={sweep}
      measFreqMhz={measFreqMhz}
      running={sweepRunning}
    />
  );
}

type FarFieldCut = "xy" | "yz";

function FarFieldChart({
  result,
  pattern,
  size,
  cut,
  azElevDeg,
  elevAzDeg,
}: {
  result: SolveResponse | null;
  pattern: PatternData | null;
  size: number;
  cut: FarFieldCut;
  azElevDeg: number;
  elevAzDeg: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const R = size / 2 - 14;

    const groundOn = !!result?.ground;
    // Azimuth cut: cone above horizon at elevation azElevDeg. With ground
    // off, the conventional setting is 0° (the xy plane). With ground on,
    // 0° is grazing and Fresnel kills the pattern, so something like 15°
    // gives a useful view — the slider lets the user pick.
    const azElevRad = (azElevDeg * Math.PI) / 180;
    const azSinT = Math.cos(azElevRad); // sin(polar θ from +z) = cos(elevation)
    const azCosT = Math.sin(azElevRad); // cos(polar θ) = sin(elevation)
    // Elevation cut: vertical great circle through azimuth bearing elevAzDeg.
    // t=0 lies at +elevAz horizon; t=π/2 is zenith; t=π is at the opposite
    // horizon; t=3π/2 is nadir (below ground, zeroed when ground is on).
    const elevAzRad = (elevAzDeg * Math.PI) / 180;
    const elevAzCos = Math.cos(elevAzRad);
    const elevAzSin = Math.sin(elevAzRad);

    // Radial axis: absolute directivity in dBi over a fixed displayable
    // range of +10 (outer edge) to −20 (origin). Labeled ticks are at the
    // multiples of 6 strictly inside that range: +6, 0, −6, −12, −18.
    const DBI_TOP = 10;
    const DB_SPAN = 30;
    const dbiToFrac = (db: number) => Math.max(0, (db - (DBI_TOP - DB_SPAN)) / DB_SPAN);
    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = 0.6;
    ctx.fillStyle = "#4a5160";
    ctx.font = "9px ui-monospace, monospace";
    for (const db of [6, 0, -6, -12, -18]) {
      const f = dbiToFrac(db);
      ctx.beginPath();
      ctx.arc(cx, cy, R * f, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillText(`${db > 0 ? "+" : ""}${db}`, cx + 2, cy - R * f - 1);
    }
    ctx.beginPath();
    ctx.moveTo(cx - R, cy);
    ctx.lineTo(cx + R, cy);
    ctx.moveTo(cx, cy - R);
    ctx.lineTo(cx, cy + R);
    ctx.stroke();

    // Axis labels: xy cut uses world x/y around the rim; yz cut shows the
    // azimuth bearing on the horizontal pair and zenith/nadir on vertical.
    ctx.fillStyle = "#4a5160";
    ctx.font = "10px ui-monospace, monospace";
    const cutLabel =
      cut === "xy"
        ? `az @ ${azElevDeg}° elev (dBi)`
        : `elev @ ${elevAzDeg}° az (dBi)`;
    ctx.fillText(cutLabel, 6, 14);
    ctx.fillStyle = "#7b8493";
    if (cut === "xy") {
      ctx.fillText("+x", cx + R - 14, cy + 11);
      ctx.fillText("−x", cx - R + 2, cy + 11);
      ctx.fillText("+y", cx - 8, cy - R + 12);
      ctx.fillText("−y", cx - 7, cy + R - 2);
    } else {
      ctx.fillText("zen", cx - 9, cy - R + 12);
      ctx.fillText("nad", cx - 9, cy + R - 2);
    }

    // Cross-reference: a single dashed spoke showing where the *other* cut
    // slices this plot. The opposite side is implied by symmetry.
    const markerStyle = "rgba(180, 140, 250, 0.7)";
    {
      const canvasAngleRad =
        cut === "xy"
          ? (elevAzDeg * Math.PI) / 180  // azimuth plot: elevation cut's bearing
          : (azElevDeg * Math.PI) / 180; // elevation plot: azimuth cut's elevation
      const cosA = Math.cos(canvasAngleRad);
      const sinA = Math.sin(canvasAngleRad);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + cosA * R, cy - sinA * R);
      ctx.strokeStyle = markerStyle;
      ctx.lineWidth = 0.8;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (!result) return;

    // Planar cut: r̂(t) = u·cos t + v·sin t, where (u, v) are the two world
    // basis vectors in the cut plane (xy: (x̂, ŷ); yz: (ŷ, ẑ)). For each
    // direction compute the moment integral over ALL wires:
    //   M(r̂) = Σ_segments I_mid · (r_{n+1} − r_n) · exp(jk r̂·r_mid)
    // and take |M_perp|² (component perpendicular to r̂).
    //
    // With a ground plane, also accumulate the PEC-image moment (segments
    // mirrored through z=0, horizontal current direction flipped) and apply
    // Fresnel coefficients per ray to get the reflected wave. Above-horizon
    // only; rays into the ground contribute nothing.
    const N_DIR = 180;
    const c = 299_792_458;
    const k = (2 * Math.PI * result.measurement_freq_mhz * 1e6) / c;
    // ε̃ = εr − j·σ/(ωε₀). Use stored constants when ground is on.
    const omega = 2 * Math.PI * result.measurement_freq_mhz * 1e6;
    const EPS0 = 8.854187817e-12;
    const epsRe = result.ground_eps_r ?? 1;
    const epsIm = -(result.ground_sigma ?? 0) / (omega * EPS0);

    // Flatten per-segment quantities across every wire.
    let nSeg = 0;
    for (const w of result.wires) nSeg += w.knot_positions.length - 1;
    const dx = new Float64Array(nSeg);
    const dy = new Float64Array(nSeg);
    const dz = new Float64Array(nSeg);
    const midx = new Float64Array(nSeg);
    const midy = new Float64Array(nSeg);
    const midz = new Float64Array(nSeg);
    const Ire = new Float64Array(nSeg);
    const Iim = new Float64Array(nSeg);
    let off = 0;
    for (const w of result.wires) {
      const knots = w.knot_positions;
      const cre = w.knot_currents_re;
      const cim = w.knot_currents_im;
      for (let n = 0; n < knots.length - 1; n++) {
        const a = knots[n];
        const b = knots[n + 1];
        dx[off] = b[0] - a[0];
        dy[off] = b[1] - a[1];
        dz[off] = b[2] - a[2];
        midx[off] = 0.5 * (a[0] + b[0]);
        midy[off] = 0.5 * (a[1] + b[1]);
        midz[off] = 0.5 * (a[2] + b[2]);
        Ire[off] = 0.5 * (cre[n] + cre[n + 1]);
        Iim[off] = 0.5 * (cim[n] + cim[n + 1]);
        off++;
      }
    }

    const mag2s = new Array<number>(N_DIR);
    let maxMag2 = 0;

    for (let pi = 0; pi < N_DIR; pi++) {
      const t = (2 * Math.PI * pi) / N_DIR;
      const ct = Math.cos(t);
      const st = Math.sin(t);
      // xy cut: cone at the chosen elevation. yz cut: vertical great circle
      // through the chosen azimuth bearing (cos t · (cos φ, sin φ) on the
      // horizontal plane, plus sin t on z).
      const rx = cut === "xy" ? azSinT * ct : elevAzCos * ct;
      const ry = cut === "xy" ? azSinT * st : elevAzSin * ct;
      const rz = cut === "xy" ? azCosT : st;

      // Rays into the ground (rz < 0) carry no far field.
      if (groundOn && rz < 0) {
        mag2s[pi] = 0;
        continue;
      }

      let mxRe = 0, mxIm = 0, myRe = 0, myIm = 0, mzRe = 0, mzIm = 0;
      // Image moment accumulators (only used when groundOn).
      let ixRe = 0, ixIm = 0, iyRe = 0, iyIm = 0, izRe = 0, izIm = 0;
      for (let n = 0; n < nSeg; n++) {
        const phase = k * (rx * midx[n] + ry * midy[n] + rz * midz[n]);
        const cph = Math.cos(phase);
        const sph = Math.sin(phase);
        // I_mid * exp(jphase)
        const ire = Ire[n] * cph - Iim[n] * sph;
        const iim = Ire[n] * sph + Iim[n] * cph;
        mxRe += ire * dx[n];
        mxIm += iim * dx[n];
        myRe += ire * dy[n];
        myIm += iim * dy[n];
        mzRe += ire * dz[n];
        mzIm += iim * dz[n];

        if (groundOn) {
          // Image position: (x, y, -z). Image current dir: (-dx, -dy, +dz).
          const phaseI = k * (rx * midx[n] + ry * midy[n] - rz * midz[n]);
          const cphI = Math.cos(phaseI);
          const sphI = Math.sin(phaseI);
          const ireI = Ire[n] * cphI - Iim[n] * sphI;
          const iimI = Ire[n] * sphI + Iim[n] * cphI;
          ixRe += ireI * -dx[n]; ixIm += iimI * -dx[n];
          iyRe += ireI * -dy[n]; iyIm += iimI * -dy[n];
          izRe += ireI *  dz[n]; izIm += iimI *  dz[n];
        }
      }
      // Direct M_perp = M − (M·r̂) r̂
      const mDotRre = mxRe * rx + myRe * ry + mzRe * rz;
      const mDotRim = mxIm * rx + myIm * ry + mzIm * rz;
      let pxRe = mxRe - mDotRre * rx;
      let pxIm = mxIm - mDotRim * rx;
      let pyRe = myRe - mDotRre * ry;
      let pyIm = myIm - mDotRim * ry;
      let pzRe = mzRe - mDotRre * rz;
      let pzIm = mzIm - mDotRim * rz;

      if (groundOn) {
        // Image M_perp.
        const iDotRre = ixRe * rx + iyRe * ry + izRe * rz;
        const iDotRim = ixIm * rx + iyIm * ry + izIm * rz;
        const qxRe = ixRe - iDotRre * rx;
        const qxIm = ixIm - iDotRim * rx;
        const qyRe = iyRe - iDotRre * ry;
        const qyIm = iyIm - iDotRim * ry;
        const qzRe = izRe - iDotRre * rz;
        const qzIm = izIm - iDotRim * rz;

        // Polarization basis at r̂. ĥ = ẑ × r̂ / |·|, v̂ = r̂ × ĥ.
        // Degenerate at the zenith (s≈0); pick arbitrary axes — both pol
        // coefficients agree there, so the choice doesn't affect the sum.
        const s = Math.sqrt(rx * rx + ry * ry);
        let hx: number, hy: number, hz: number;
        let vx: number, vy: number, vz: number;
        if (s > 1e-9) {
          hx = -ry / s; hy = rx / s; hz = 0;
          vx = -rx * rz / s; vy = -ry * rz / s; vz = s;
        } else {
          hx = 1; hy = 0; hz = 0;
          vx = 0; vy = 1; vz = 0;
        }

        // Decompose image perp onto (ĥ, v̂) — complex scalars.
        const qhRe = qxRe * hx + qyRe * hy + qzRe * hz;
        const qhIm = qxIm * hx + qyIm * hy + qzIm * hz;
        const qvRe = qxRe * vx + qyRe * vy + qzRe * vz;
        const qvIm = qxIm * vx + qyIm * vy + qzIm * vz;

        // Fresnel reflection coefficients (complex). cos θᵢ = rz, sin²θᵢ = s².
        // ε̃ − sin²θᵢ is complex; sqrt of complex follows the principal branch.
        const cosTi = rz;
        const sin2Ti = s * s;
        const aRe = epsRe - sin2Ti;
        const aIm = epsIm;
        // Principal-branch √(a + jb)
        const aMag = Math.hypot(aRe, aIm);
        const QRe = Math.sqrt(0.5 * (aMag + aRe));
        const QIm = Math.sign(aIm || 1) * Math.sqrt(Math.max(0, 0.5 * (aMag - aRe)));
        // ρ_h = (cosTi − Q) / (cosTi + Q)
        const numHRe = cosTi - QRe, numHIm = -QIm;
        const denHRe = cosTi + QRe, denHIm = QIm;
        const denH2 = denHRe * denHRe + denHIm * denHIm;
        const rhoHRe = (numHRe * denHRe + numHIm * denHIm) / denH2;
        const rhoHIm = (numHIm * denHRe - numHRe * denHIm) / denH2;
        // ρ_v = (ε̃·cosTi − Q) / (ε̃·cosTi + Q)
        const ecRe = epsRe * cosTi, ecIm = epsIm * cosTi;
        const numVRe = ecRe - QRe, numVIm = ecIm - QIm;
        const denVRe = ecRe + QRe, denVIm = ecIm + QIm;
        const denV2 = denVRe * denVRe + denVIm * denVIm;
        const rhoVRe = (numVRe * denVRe + numVIm * denVIm) / denV2;
        const rhoVIm = (numVIm * denVRe - numVRe * denVIm) / denV2;

        // Reflected: M_refl = ρ_v · q_v · v̂ − ρ_h · q_h · ĥ.
        // The (−ρ_h) sign folds the PEC image's pre-applied horizontal flip
        // back out, so ρ_h=−1 reproduces the PEC reflection exactly.
        const rvqRe = rhoVRe * qvRe - rhoVIm * qvIm;
        const rvqIm = rhoVRe * qvIm + rhoVIm * qvRe;
        const rhqRe = rhoHRe * qhRe - rhoHIm * qhIm;
        const rhqIm = rhoHRe * qhIm + rhoHIm * qhRe;
        pxRe += rvqRe * vx - rhqRe * hx;
        pxIm += rvqIm * vx - rhqIm * hx;
        pyRe += rvqRe * vy - rhqRe * hy;
        pyIm += rvqIm * vy - rhqIm * hy;
        pzRe += rvqRe * vz - rhqRe * hz;
        pzIm += rvqIm * vz - rhqIm * hz;
      }

      const mag2 =
        pxRe * pxRe + pxIm * pxIm +
        pyRe * pyRe + pyIm * pyIm +
        pzRe * pzRe + pzIm * pzIm;
      mag2s[pi] = mag2;
      if (mag2 > maxMag2) maxMag2 = mag2;
    }

    if (maxMag2 <= 0) return;

    // Absolute directivity: D(φ) = directivity_norm · |M_perp(π/2, φ)|².
    // If the server omitted the norm (older response), fall back to a
    // per-frame relative scale that puts the peak at 0 dBi.
    const norm =
      result.directivity_norm && result.directivity_norm > 0
        ? result.directivity_norm
        : 1 / maxMag2;

    ctx.beginPath();
    for (let pi = 0; pi <= N_DIR; pi++) {
      const t = (2 * Math.PI * pi) / N_DIR;
      const D = norm * mag2s[pi % N_DIR];
      const dBi = D > 0 ? 10 * Math.log10(D) : -Infinity;
      const frac = dbiToFrac(dBi);
      const px = cx + Math.cos(t) * frac * R;
      // Canvas y flips: +y on canvas is down, so we negate to put +y at top.
      const py = cy - Math.sin(t) * frac * R;
      if (pi === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(255, 209, 102, 0.12)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 209, 102, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // NEC exact-pattern overlay (dashed cyan line) when available. Bilinear
    // interpolation off the (θ, φ) grid; rays below horizon are skipped so
    // the line breaks at the ground rather than wrapping to the origin.
    if (pattern) {
      const nt = pattern.theta_deg.length;
      const np_ = pattern.phi_deg.length;
      const dTheta = pattern.theta_deg[1] - pattern.theta_deg[0];
      const dPhi = pattern.phi_deg[1] - pattern.phi_deg[0];
      const clip = (g: number) => (g < -100 ? -100 : g);

      ctx.beginPath();
      let started = false;
      for (let pi = 0; pi <= N_DIR; pi++) {
        const t = (2 * Math.PI * pi) / N_DIR;
        const ct = Math.cos(t);
        const st = Math.sin(t);
        const rx = cut === "xy" ? azSinT * ct : elevAzCos * ct;
        const ry = cut === "xy" ? azSinT * st : elevAzSin * ct;
        const rz = cut === "xy" ? azCosT : st;
        if (rz < -1e-9) { started = false; continue; }

        const thetaDeg = (Math.acos(Math.max(-1, Math.min(1, rz))) * 180) / Math.PI;
        let phiRad = Math.atan2(ry, rx);
        if (phiRad < 0) phiRad += 2 * Math.PI;
        const phiDeg = (phiRad * 180) / Math.PI;

        const tf = Math.max(0, Math.min(nt - 1, thetaDeg / dTheta));
        const pf = Math.max(0, Math.min(np_ - 1, phiDeg / dPhi));
        const t0 = Math.floor(tf), t1 = Math.min(nt - 1, t0 + 1);
        const p0 = Math.floor(pf), p1 = Math.min(np_ - 1, p0 + 1);
        const ft = tf - t0, fp = pf - p0;
        const g00 = clip(pattern.gain_dbi[t0][p0]);
        const g01 = clip(pattern.gain_dbi[t0][p1]);
        const g10 = clip(pattern.gain_dbi[t1][p0]);
        const g11 = clip(pattern.gain_dbi[t1][p1]);
        const dBi =
          g00 * (1 - ft) * (1 - fp) +
          g01 * (1 - ft) * fp +
          g10 * ft * (1 - fp) +
          g11 * ft * fp;

        const frac = dbiToFrac(dBi);
        const px = cx + Math.cos(t) * frac * R;
        const py = cy - Math.sin(t) * frac * R;
        if (!started) { ctx.moveTo(px, py); started = true; }
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = "rgba(110, 220, 255, 0.85)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Legend swatch + label, bottom-right.
      ctx.fillStyle = "rgba(110, 220, 255, 0.9)";
      ctx.font = "10px ui-monospace, monospace";
      const necText = "NEC rp_card";
      const necTw = ctx.measureText(necText).width;
      ctx.fillText(necText, size - necTw - 6, size - 6);
    }

    // Peak dBi annotation (top-right corner).
    const peakDbi = 10 * Math.log10(norm * maxMag2);
    ctx.fillStyle = "#cdd5e0";
    ctx.font = "10px ui-monospace, monospace";
    const peakText = `peak ${peakDbi >= 0 ? "+" : ""}${peakDbi.toFixed(1)} dBi`;
    const tw = ctx.measureText(peakText).width;
    ctx.fillText(peakText, size - tw - 6, 14);
  }, [result, pattern, size, cut, azElevDeg, elevAzDeg]);

  return <canvas ref={canvasRef} className="farfield" />;
}

function SmithChart({
  r,
  x,
  z0,
  size,
  sweep,
  measFreqMhz,
  running,
}: {
  r: number;
  x: number;
  z0: number;
  size: number;
  sweep: SweepData | null;
  measFreqMhz: number;
  running: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cx = size / 2;
    const cy = size / 2;
    const R = size / 2 - 10;

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    // Constant-r circles in the Γ plane.
    // Each maps to a circle: center = (r/(r+1), 0), radius = 1/(r+1).
    const rCircles: { r: number; label?: string }[] = [
      { r: 0.2 },
      { r: 0.5, label: "0.5" },
      { r: 1, label: "1" },
      { r: 2, label: "2" },
      { r: 5 },
    ];
    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = 0.6;
    for (const { r: rn } of rCircles) {
      const cxN = rn / (rn + 1);
      const radN = 1 / (rn + 1);
      ctx.beginPath();
      ctx.arc(cx + cxN * R, cy, radN * R, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Constant-x arcs: center = (1, 1/x), radius = 1/|x|. Clip to unit disk.
    const xArcs = [0.2, 0.5, 1, 2, 5];
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.clip();
    for (const xn of xArcs) {
      const arcCx = cx + R;
      const rad = (1 / xn) * R;
      // Inductive (X > 0)
      ctx.beginPath();
      ctx.arc(arcCx, cy - (1 / xn) * R, rad, 0, 2 * Math.PI);
      ctx.stroke();
      // Capacitive (X < 0)
      ctx.beginPath();
      ctx.arc(arcCx, cy + (1 / xn) * R, rad, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.restore();

    // Real axis
    ctx.strokeStyle = "#3a4150";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(cx - R, cy);
    ctx.lineTo(cx + R, cy);
    ctx.stroke();

    // Outer boundary (|Γ| = 1)
    ctx.strokeStyle = "#3a4150";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.stroke();

    // Z0 label at center
    ctx.fillStyle = "#4a5160";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`Z₀ = ${z0}`, 6, 14);

    // Reactance sign labels.
    ctx.fillStyle = "#4a5160";
    ctx.fillText("+jX", cx + R - 24, cy - R + 14);
    ctx.fillText("−jX", cx + R - 24, cy + R - 4);

    // Sweep locus: blue points at each Γ-plane sample (no connecting line —
    // sparse samples make a piecewise polyline read as artificial kinks).
    if (sweep && sweep.freqs_mhz.length > 1) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, 2 * Math.PI);
      ctx.clip();
      ctx.fillStyle = "rgba(118, 208, 255, 0.85)";
      let nearestIdx = 0;
      let nearestDelta = Infinity;
      for (let i = 0; i < sweep.freqs_mhz.length; i++) {
        const g = reflectionCoefficient(sweep.z_re[i], sweep.z_im[i], z0);
        const px = cx + g.gRe * R;
        const py = cy - g.gIm * R;
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, 2 * Math.PI);
        ctx.fill();
        const d = Math.abs(sweep.freqs_mhz[i] - measFreqMhz);
        if (d < nearestDelta) {
          nearestDelta = d;
          nearestIdx = i;
        }
      }
      ctx.restore();

      // Endpoint markers (low-freq filled, high-freq hollow).
      const drawEndpoint = (idx: number, filled: boolean) => {
        const g = reflectionCoefficient(sweep.z_re[idx], sweep.z_im[idx], z0);
        const px = cx + g.gRe * R;
        const py = cy - g.gIm * R;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = "rgba(118, 208, 255, 0.95)";
        ctx.fillStyle = filled ? "rgba(118, 208, 255, 0.95)" : "rgba(13, 16, 21, 0.95)";
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      };
      drawEndpoint(0, true);
      drawEndpoint(sweep.freqs_mhz.length - 1, false);

      // Freq range label across the bottom of the panel.
      ctx.fillStyle = "#9aa3b2";
      ctx.font = "10px ui-monospace, monospace";
      const fLoTxt = sweep.freqs_mhz[0].toFixed(2);
      const fHiTxt = sweep.freqs_mhz[sweep.freqs_mhz.length - 1].toFixed(2);
      const txt = `${fLoTxt} → ${fHiTxt} MHz`;
      ctx.fillText(txt, size - 6 - ctx.measureText(txt).width, size - 6);

      // Tick which sweep sample matches the current meas freq (if any).
      void nearestIdx;
    }

    if (running) {
      ctx.fillStyle = "#7b8493";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText("sweeping…", 6, size - 6);
    }

    // Current impedance marker.
    if (r > 0 || x !== 0) {
      const { gRe, gIm } = reflectionCoefficient(r, x, z0);
      // gIm > 0 means inductive (top half); canvas y flips.
      const px = cx + gRe * R;
      const py = cy - gIm * R;

      // Line from center
      ctx.strokeStyle = "rgba(255, 209, 102, 0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(px, py);
      ctx.stroke();

      // Glow
      const grad = ctx.createRadialGradient(px, py, 0, px, py, 14);
      grad.addColorStop(0, "rgba(255, 209, 102, 0.55)");
      grad.addColorStop(1, "rgba(255, 209, 102, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, 14, 0, 2 * Math.PI);
      ctx.fill();

      // Dot
      ctx.fillStyle = "#ffd166";
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Center match marker
    ctx.strokeStyle = "#5a6170";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 4, cy);
    ctx.lineTo(cx + 4, cy);
    ctx.moveTo(cx, cy - 4);
    ctx.lineTo(cx, cy + 4);
    ctx.stroke();
  }, [r, x, z0, size, sweep, measFreqMhz, running]);

  return <canvas ref={canvasRef} className="smith" />;
}

function CurrentCanvas({
  result,
  projection,
  showHeatmap,
  showEnvelope,
}: {
  result: SolveResponse | null;
  projection: Projection;
  showHeatmap: boolean;
  showEnvelope: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const onResize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    };

    function draw() {
      if (!canvas) return;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx!.clearRect(0, 0, w, h);

      // Vertical axis guide.
      ctx!.strokeStyle = "#23272f";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(w / 2, 20);
      ctx!.lineTo(w / 2, h - 20);
      ctx!.stroke();

      if (!result) return;

      // Scale anchored to design wavelength. Worst-case extents (in λ):
      //   horizontal: hf_max × λ/2 ≈ 0.6 λ  (both V and Yagi)
      //   vertical:   max(V droop, Yagi spacing) ≈ 0.5 λ
      //
      // `s` proportionally shrinks every fixed pixel constant (padding,
      // strokes, envelope amplitude, label sizes) so the rendering looks
      // the same at thumbnail and main sizes. Floor keeps thumbnails
      // legible; cap prevents very-large canvases from over-inflating.
      const refSize = 600;
      const s = Math.max(0.3, Math.min(1.4, Math.min(w, h) / refSize));
      const C_LIGHT = 299_792_458.0;
      const lambdaDesign = C_LIGHT / (result.design_freq_mhz * 1e6);
      const pad = 50 * s;
      const barReserveBottom = 40 * s;
      const FILL = 0.85;
      const scale = FILL * Math.min(
        (w - 2 * pad) / (0.6 * lambdaDesign),
        (h - pad - barReserveBottom) / (0.5 * lambdaDesign),
      );

      // Camera projection: pick the two world axes to map to canvas
      // (horizontal, vertical). The hidden axis is the camera ray. App.tsx
      // sets a per-geometry default (V/fan_dipole → "yz" side, Yagi/moxon/
      // hexbeam → "xy" top) but the user can override via the projection
      // toggle in the stage.
      const projSpec = PROJECTIONS.find((p) => p.id === projection)!;
      const horizAxis = projSpec.horizAxis;
      const vertAxis = projSpec.vertAxis;
      let hMin = Infinity, hMax = -Infinity;
      let vMin = Infinity, vMax = -Infinity;
      for (const wire of result.wires) {
        for (const p of wire.knot_positions) {
          if (p[horizAxis] < hMin) hMin = p[horizAxis];
          if (p[horizAxis] > hMax) hMax = p[horizAxis];
          if (p[vertAxis] < vMin) vMin = p[vertAxis];
          if (p[vertAxis] > vMax) vMax = p[vertAxis];
        }
      }
      const hC = (hMin + hMax) / 2;
      const vC = (vMin + vMax) / 2;
      const cx = w / 2;
      const cy = h / 2;
      const project = (p: [number, number, number]) => ({
        x: cx + (p[horizAxis] - hC) * scale,
        y: cy + (vC - p[vertAxis]) * scale, // higher vert value = higher on screen
      });

      // Global current magnitude so the per-wire colors share a scale.
      let magMaxGlobal = 1e-30;
      const perWireMags: number[][] = [];
      for (const wire of result.wires) {
        const m = wire.knot_currents_re.map((r, i) =>
          Math.hypot(r, wire.knot_currents_im[i]),
        );
        perWireMags.push(m);
        for (const v of m) if (v > magMaxGlobal) magMaxGlobal = v;
      }

      ctx!.lineCap = "round";
      ctx!.lineJoin = "round";

      // One wire at a time: wire stroke + envelope.
      const envScale = 60 * s;
      const labelFontPx = Math.max(8, Math.round(11 * s));
      const feedFontPx = Math.max(8, Math.round(12 * s));
      const feedWireIdx = result.feed_wire_index;
      for (let wi = 0; wi < result.wires.length; wi++) {
        const wire = result.wires[wi];
        const knots = wire.knot_positions;
        const mags = perWireMags[wi];

        for (let i = 0; i < knots.length - 1; i++) {
          const a = project(knots[i]);
          const b = project(knots[i + 1]);
          if (showHeatmap) {
            const m = (0.5 * (mags[i] + mags[i + 1])) / magMaxGlobal;
            ctx!.strokeStyle = currentColor(m);
            ctx!.lineWidth = (2 + 6 * m) * s;
          } else {
            // Plain wires: uniform color/width, no current-magnitude modulation.
            ctx!.strokeStyle = "#9aa3b2";
            ctx!.lineWidth = 2 * s;
          }
          ctx!.beginPath();
          ctx!.moveTo(a.x, a.y);
          ctx!.lineTo(b.x, b.y);
          ctx!.stroke();
        }

        // Current-waveform envelope: if this is the feed wire (and the feed
        // isn't at an end), split at the feed knot so a V's per-arm tangent
        // flip is respected. Otherwise draw one continuous envelope.
        if (showEnvelope) {
          ctx!.strokeStyle = "rgba(118, 208, 255, 0.7)";
          ctx!.lineWidth = 1.5 * s;
          const lastIdx = knots.length - 1;
          const feedIdx = result.feed_knot_index;
          if (wi === feedWireIdx && feedIdx > 0 && feedIdx < lastIdx) {
            drawArmEnvelope(ctx!, knots, mags, magMaxGlobal, project, 0, feedIdx, envScale);
            drawArmEnvelope(ctx!, knots, mags, magMaxGlobal, project, feedIdx, lastIdx, envScale);
          } else {
            drawArmEnvelope(ctx!, knots, mags, magMaxGlobal, project, 0, lastIdx, envScale);
          }
        }

        // Wire label near the leftmost knot for multi-wire geometries.
        if (result.wires.length > 1) {
          const lp = project(knots[0]);
          ctx!.fillStyle = "#7b8493";
          ctx!.font = `${labelFontPx}px ui-monospace, monospace`;
          ctx!.fillText(wire.label, lp.x - 8 * s - ctx!.measureText(wire.label).width, lp.y + 3 * s);
        }
      }

      // Feed marker on the feed wire.
      const feedWire = result.wires[feedWireIdx];
      if (feedWire) {
        const feed = project(feedWire.knot_positions[result.feed_knot_index]);
        ctx!.fillStyle = "#ffd166";
        ctx!.beginPath();
        ctx!.arc(feed.x, feed.y, 5 * s, 0, Math.PI * 2);
        ctx!.fill();
        ctx!.font = `${feedFontPx}px ui-monospace, monospace`;
        ctx!.fillText("feed", feed.x + 8 * s, feed.y - 8 * s);
      }

      // λ/4 scale bar, centered horizontally under the antenna.
      const barLenPx = (lambdaDesign / 4) * scale;
      const barX0 = (w - barLenPx) / 2;
      const barY = h - 24 * s;
      ctx!.strokeStyle = "#7b8493";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(barX0, barY);
      ctx!.lineTo(barX0 + barLenPx, barY);
      ctx!.moveTo(barX0, barY - 4 * s);
      ctx!.lineTo(barX0, barY + 4 * s);
      ctx!.moveTo(barX0 + barLenPx, barY - 4 * s);
      ctx!.lineTo(barX0 + barLenPx, barY + 4 * s);
      ctx!.stroke();
      ctx!.fillStyle = "#9aa3b2";
      ctx!.font = `${labelFontPx}px ui-monospace, monospace`;
      const barLabel = `λ/4 = ${(lambdaDesign / 4).toFixed(2)} m`;
      const labelW = ctx!.measureText(barLabel).width;
      ctx!.fillText(barLabel, (w - labelW) / 2, barY - 8 * s);
    }

    onResize();
    const obs = new ResizeObserver(onResize);
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [result, projection, showHeatmap, showEnvelope]);

  return <canvas ref={canvasRef} />;
}

function drawArmEnvelope(
  ctx: CanvasRenderingContext2D,
  knots: [number, number, number][],
  mags: number[],
  magMax: number,
  project: (p: [number, number, number]) => { x: number; y: number },
  start: number,
  end: number,
  envScale: number,
) {
  if (end <= start) return;

  // Per-segment normal in canvas space, oriented toward screen-up so V-style
  // arms put their envelopes "above" the wire. For axis-aligned vertical
  // segments ny is exactly zero and the flip is a no-op; that's fine — what
  // matters is that the moxon's adjacent perpendicular segments get
  // *different* normals so the bend-break below catches the corner.
  const segN: { nx: number; ny: number }[] = [];
  for (let i = start; i < end; i++) {
    const p = project(knots[i]);
    const q = project(knots[i + 1]);
    const dx = q.x - p.x;
    const dy = q.y - p.y;
    const len = Math.hypot(dx, dy) || 1;
    let nx = -dy / len;
    let ny = dx / len;
    if (ny > 0) {
      nx = -nx;
      ny = -ny;
    }
    segN.push({ nx, ny });
  }

  // Walk runs of segments whose normals agree (within ~3°), and start a new
  // sub-path at each bend. Without this, a connected envelope at a 90°
  // corner zigzags across the corner since the two adjacent segments offset
  // their knots in perpendicular directions.
  const BEND_TOL = 0.9986;  // cos(3°)
  ctx.beginPath();
  let s = 0;
  while (s < segN.length) {
    let e = s;
    while (
      e + 1 < segN.length &&
      segN[e].nx * segN[e + 1].nx + segN[e].ny * segN[e + 1].ny >= BEND_TOL
    ) {
      e++;
    }
    const { nx, ny } = segN[s];
    for (let k = s; k <= e + 1; k++) {
      const ki = start + k;
      const p = project(knots[ki]);
      const offset = (mags[ki] / magMax) * envScale;
      const ex = p.x + nx * offset;
      const ey = p.y + ny * offset;
      if (k === s) ctx.moveTo(ex, ey);
      else ctx.lineTo(ex, ey);
    }
    s = e + 1;
  }
  ctx.stroke();
}

function currentColor(t: number): string {
  // Cool → warm ramp: dim blue → cyan → yellow → orange.
  const stops = [
    [0.0, [40, 64, 96]],
    [0.25, [60, 140, 200]],
    [0.5, [118, 208, 255]],
    [0.75, [255, 209, 102]],
    [1.0, [255, 130, 80]],
  ] as const;
  for (let i = 1; i < stops.length; i++) {
    const [t0, c0] = stops[i - 1];
    const [t1, c1] = stops[i];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0 || 1);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(255,130,80)";
}
