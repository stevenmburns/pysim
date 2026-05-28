import { useEffect, useRef, useState } from "react";

type Wire = {
  label: string;
  knot_positions: [number, number, number][];
  knot_currents_re: number[];
  knot_currents_im: number[];
};

type SolveResponse = {
  geometry: "inverted_v" | "yagi";
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
  // V-specific
  arm_len_m?: number;
  // Yagi-specific
  driver_length_m?: number;
  reflector_length_m?: number;
  spacing_m?: number;
};

type SolveRequest = {
  geometry: "inverted_v" | "yagi";
  solver: "pysim" | "pynec";
  n_per_wire: number;
  design_freq_mhz: number;
  measurement_freq_mhz: number;
  wire_radius: number;
  // V
  angle_deg?: number;
  halfdriver_factor?: number;
  // Yagi
  driver_length_factor?: number;
  reflector_length_factor?: number;
  spacing_wavelengths?: number;
};

type SweepData = {
  freqs_mhz: number[];
  z_re: number[];
  z_im: number[];
};

const WS_URL = `ws://${window.location.host}/ws`;

export function App() {
  const [geometry, setGeometry] = useState<"inverted_v" | "yagi">("inverted_v");
  // V controls
  const [angle, setAngle] = useState(30);
  const [halfdriverFactor, setHalfdriverFactor] = useState(0.962);
  // Yagi controls
  const [driverLengthFactor, setDriverLengthFactor] = useState(0.962);
  const [reflectorLengthFactor, setReflectorLengthFactor] = useState(1.01);
  const [spacingWavelengths, setSpacingWavelengths] = useState(0.15);
  // Shared
  const [solver, setSolver] = useState<"pysim" | "pynec">("pysim");
  const [nPerWire, setNPerWire] = useState(30);
  const [designFreq, setDesignFreq] = useState(14.3);
  const [measFreq, setMeasFreq] = useState(14.3);
  const [linkMeas, setLinkMeas] = useState(true);
  const [wireRadius, setWireRadius] = useState(0.0005);

  // When linked, design and measurement freq move together.
  function updateDesignFreq(v: number) {
    setDesignFreq(v);
    if (linkMeas) setMeasFreq(v);
  }
  function toggleLink(next: boolean) {
    setLinkMeas(next);
    if (next) setMeasFreq(designFreq);
  }

  const [result, setResult] = useState<SolveResponse | null>(null);
  const [status, setStatus] = useState<"connecting" | "open" | "closed">("connecting");
  const [rttMs, setRttMs] = useState<number | null>(null);
  const [sweep, setSweep] = useState<SweepData | null>(null);
  const [sweepRunning, setSweepRunning] = useState(false);

  const sweepTimerRef = useRef<number | null>(null);
  const sweepAbortRef = useRef<AbortController | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const inFlightRef = useRef(false);
  const pendingRef = useRef<SolveRequest | null>(null);
  const sendStartRef = useRef(0);

  function buildRequest(): SolveRequest {
    const base: SolveRequest = {
      geometry,
      solver,
      n_per_wire: nPerWire,
      design_freq_mhz: designFreq,
      measurement_freq_mhz: measFreq,
      wire_radius: wireRadius,
    };
    if (geometry === "inverted_v") {
      base.angle_deg = angle;
      base.halfdriver_factor = halfdriverFactor;
    } else {
      base.driver_length_factor = driverLengthFactor;
      base.reflector_length_factor = reflectorLengthFactor;
      base.spacing_wavelengths = spacingWavelengths;
    }
    return base;
  }

  // The latest control values, used to send a new request when the prior one
  // completes (drops intermediate values rather than queuing them all up).
  const controlsRef = useRef<SolveRequest>(buildRequest());

  useEffect(() => {
    controlsRef.current = buildRequest();
    requestSolve();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    geometry, solver,
    angle, halfdriverFactor,
    driverLengthFactor, reflectorLengthFactor, spacingWavelengths,
    nPerWire, designFreq, measFreq, wireRadius,
  ]);

  // Debounced sweep across measurement freq. Re-runs whenever any antenna
  // parameter changes; measurement-freq slider does NOT trigger a sweep.
  useEffect(() => {
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
    geometry, solver,
    angle, halfdriverFactor,
    driverLengthFactor, reflectorLengthFactor, spacingWavelengths,
    nPerWire, designFreq, wireRadius,
  ]);

  async function runSweep() {
    sweepAbortRef.current?.abort();
    const controller = new AbortController();
    sweepAbortRef.current = controller;

    // Sweep 0.8x to 1.25x of design freq with 41 log-spaced points.
    const N = 41;
    const fLo = Math.max(0.5, designFreq * 0.8);
    const fHi = Math.min(60, designFreq * 1.25);
    const freqs = Array.from({ length: N }, (_, i) =>
      Math.exp(Math.log(fLo) + (i / (N - 1)) * (Math.log(fHi) - Math.log(fLo))),
    );

    const body = { ...buildRequest(), freqs_mhz: freqs };
    setSweepRunning(true);
    try {
      const resp = await fetch("/sweep", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error(`sweep failed: ${resp.status}`);
      const data: SweepData = await resp.json();
      if (!controller.signal.aborted) setSweep(data);
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
      pendingRef.current = controlsRef.current;
      requestSolve();
    };
    ws.onclose = () => setStatus("closed");
    ws.onerror = () => setStatus("closed");
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

        <div className="geometry-tabs" role="tablist">
          <button
            role="tab"
            aria-selected={geometry === "inverted_v"}
            className={geometry === "inverted_v" ? "active" : ""}
            onClick={() => setGeometry("inverted_v")}
          >
            inverted V
          </button>
          <button
            role="tab"
            aria-selected={geometry === "yagi"}
            className={geometry === "yagi" ? "active" : ""}
            onClick={() => setGeometry("yagi")}
          >
            Yagi (2 elements)
          </button>
        </div>

        <div className="group-label">antenna</div>

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
          </>
        )}

        <div className="field">
          <label>
            <span>design freq</span>
            <span>{designFreq.toFixed(3)} MHz</span>
          </label>
          <input
            type="range"
            min={11.44}
            max={17.875}
            step={0.005}
            value={designFreq}
            onInput={(e) => updateDesignFreq(Number((e.target as HTMLInputElement).value))}
          />
        </div>

        <div className="field">
          <label>
            <span>wire radius (m)</span>
          </label>
          <input
            type="number"
            step={0.0001}
            value={wireRadius}
            onChange={(e) => setWireRadius(Number(e.target.value) || 0)}
          />
        </div>

        <div className="group-label">simulation</div>

        <div className="field">
          <label>
            <span>solver</span>
            <span>{solver}</span>
          </label>
          <div className="geometry-tabs" role="tablist">
            <button
              role="tab"
              aria-selected={solver === "pysim"}
              className={solver === "pysim" ? "active" : ""}
              onClick={() => setSolver("pysim")}
            >
              pysim
            </button>
            <button
              role="tab"
              aria-selected={solver === "pynec"}
              className={solver === "pynec" ? "active" : ""}
              onClick={() => setSolver("pynec")}
            >
              PyNEC
            </button>
          </div>
        </div>

        <div className="field">
          <label>
            <span>segments / wire (N)</span>
            <span>{nPerWire}</span>
          </label>
          <input
            type="range"
            min={10}
            max={80}
            step={1}
            value={nPerWire}
            onInput={(e) => setNPerWire(Number((e.target as HTMLInputElement).value))}
          />
        </div>

        <div className="field">
          <label>
            <span>measurement freq</span>
            <span>{measFreq.toFixed(3)} MHz</span>
          </label>
          <input
            type="range"
            min={Math.max(0.5, designFreq * 0.8)}
            max={Math.min(60, designFreq * 1.25)}
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
      </aside>

      <main className="stage">
        <CurrentCanvas result={result} />
        <div className="farfield-panel">
          <FarFieldChart result={result} size={220} cut="xy" />
          <FarFieldChart result={result} size={220} cut="yz" />
        </div>
        <div className="smith-panel">
          <SmithChart
            r={result?.z_in_re ?? 0}
            x={result?.z_in_im ?? 0}
            z0={50}
            size={260}
            sweep={sweep}
            measFreqMhz={measFreq}
            running={sweepRunning}
          />
        </div>
        <div className="status">ws: {status}</div>
      </main>
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

type FarFieldCut = "xy" | "yz";

function FarFieldChart({
  result,
  size,
  cut,
}: {
  result: SolveResponse | null;
  size: number;
  cut: FarFieldCut;
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

    // Radial axis: absolute directivity in dBi. Outer ring = DBI_TOP,
    // floor = DBI_TOP − DB_SPAN. Rings every 10 dB.
    const DBI_TOP = 10;
    const DB_SPAN = 30;
    const dbiToFrac = (db: number) => Math.max(0, (db - (DBI_TOP - DB_SPAN)) / DB_SPAN);
    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = 0.6;
    ctx.fillStyle = "#4a5160";
    ctx.font = "9px ui-monospace, monospace";
    for (let db = DBI_TOP - DB_SPAN + 10; db <= DBI_TOP; db += 10) {
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

    // Axis labels parameterized by cut plane. The "horizontal" axis on the
    // canvas is whichever world axis lies in the cut and serves as the
    // 0-angle reference (cos t); "vertical" is the second cut axis (sin t).
    const horizLabel = cut === "xy" ? "x" : "y";
    const vertLabel = cut === "xy" ? "y" : "z";
    ctx.fillStyle = "#4a5160";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`${cut} plane (dBi)`, 6, 14);
    ctx.fillStyle = "#7b8493";
    ctx.fillText(`+${horizLabel}`, cx + R - 14, cy + 11);
    ctx.fillText(`−${horizLabel}`, cx - R + 2, cy + 11);
    ctx.fillText(`+${vertLabel}`, cx - 8, cy - R + 12);
    ctx.fillText(`−${vertLabel}`, cx - 7, cy + R - 2);

    if (!result) return;

    // Planar cut: r̂(t) = u·cos t + v·sin t, where (u, v) are the two world
    // basis vectors in the cut plane (xy: (x̂, ŷ); yz: (ŷ, ẑ)). For each
    // direction compute the moment integral over ALL wires:
    //   M(r̂) = Σ_segments I_mid · (r_{n+1} − r_n) · exp(jk r̂·r_mid)
    // and take |M_perp|² (component perpendicular to r̂).
    const N_DIR = 180;
    const c = 299_792_458;
    const k = (2 * Math.PI * result.measurement_freq_mhz * 1e6) / c;

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
      // (u, v) = (x̂, ŷ) for xy cut, (ŷ, ẑ) for yz cut.
      const rx = cut === "xy" ? ct : 0;
      const ry = cut === "xy" ? st : ct;
      const rz = cut === "xy" ? 0 : st;

      let mxRe = 0;
      let mxIm = 0;
      let myRe = 0;
      let myIm = 0;
      let mzRe = 0;
      let mzIm = 0;
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
      }
      // M·r̂ uses all three components now.
      const mDotRre = mxRe * rx + myRe * ry + mzRe * rz;
      const mDotRim = mxIm * rx + myIm * ry + mzIm * rz;
      // M_perp = M − (M·r̂) r̂
      const pxRe = mxRe - mDotRre * rx;
      const pxIm = mxIm - mDotRim * rx;
      const pyRe = myRe - mDotRre * ry;
      const pyIm = myIm - mDotRim * ry;
      const pzRe = mzRe - mDotRre * rz;
      const pzIm = mzIm - mDotRim * rz;
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

    // Peak dBi annotation (top-right corner).
    const peakDbi = 10 * Math.log10(norm * maxMag2);
    ctx.fillStyle = "#cdd5e0";
    ctx.font = "10px ui-monospace, monospace";
    const peakText = `peak ${peakDbi >= 0 ? "+" : ""}${peakDbi.toFixed(1)} dBi`;
    const tw = ctx.measureText(peakText).width;
    ctx.fillText(peakText, size - tw - 6, 14);
  }, [result, size, cut]);

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

    // Sweep locus: continuous curve through Γ-plane samples.
    if (sweep && sweep.freqs_mhz.length > 1) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, 2 * Math.PI);
      ctx.clip();
      ctx.strokeStyle = "rgba(118, 208, 255, 0.75)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let nearestIdx = 0;
      let nearestDelta = Infinity;
      for (let i = 0; i < sweep.freqs_mhz.length; i++) {
        const g = reflectionCoefficient(sweep.z_re[i], sweep.z_im[i], z0);
        const px = cx + g.gRe * R;
        const py = cy - g.gIm * R;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
        const d = Math.abs(sweep.freqs_mhz[i] - measFreqMhz);
        if (d < nearestDelta) {
          nearestDelta = d;
          nearestIdx = i;
        }
      }
      ctx.stroke();
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

function CurrentCanvas({ result }: { result: SolveResponse | null }) {
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
      const C_LIGHT = 299_792_458.0;
      const lambdaDesign = C_LIGHT / (result.design_freq_mhz * 1e6);
      const pad = 50;
      const barReserveBottom = 40;
      const FILL = 0.85;
      const scale = FILL * Math.min(
        (w - 2 * pad) / (0.6 * lambdaDesign),
        (h - pad - barReserveBottom) / (0.5 * lambdaDesign),
      );

      // Per-geometry view: V is a side view (xz plane), Yagi is top-down
      // (xy plane). vertAxis is the world axis that maps to canvas-y.
      const vertAxis = result.geometry === "yagi" ? 1 : 2;
      let xMin = Infinity, xMax = -Infinity;
      let vMin = Infinity, vMax = -Infinity;
      for (const wire of result.wires) {
        for (const p of wire.knot_positions) {
          if (p[0] < xMin) xMin = p[0];
          if (p[0] > xMax) xMax = p[0];
          if (p[vertAxis] < vMin) vMin = p[vertAxis];
          if (p[vertAxis] > vMax) vMax = p[vertAxis];
        }
      }
      const xC = (xMin + xMax) / 2;
      const vC = (vMin + vMax) / 2;
      const cx = w / 2;
      const cy = h / 2;
      const project = (p: [number, number, number]) => ({
        x: cx + (p[0] - xC) * scale,
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
      const envScale = 60;
      const feedWireIdx = result.feed_wire_index;
      for (let wi = 0; wi < result.wires.length; wi++) {
        const wire = result.wires[wi];
        const knots = wire.knot_positions;
        const mags = perWireMags[wi];

        for (let i = 0; i < knots.length - 1; i++) {
          const a = project(knots[i]);
          const b = project(knots[i + 1]);
          const m = (0.5 * (mags[i] + mags[i + 1])) / magMaxGlobal;
          ctx!.strokeStyle = currentColor(m);
          ctx!.lineWidth = 2 + 6 * m;
          ctx!.beginPath();
          ctx!.moveTo(a.x, a.y);
          ctx!.lineTo(b.x, b.y);
          ctx!.stroke();
        }

        // Envelope: if this is the feed wire (and the feed isn't at an end),
        // split at the feed knot so a V's per-arm tangent flip is respected.
        // Otherwise draw one continuous envelope.
        ctx!.strokeStyle = "rgba(118, 208, 255, 0.7)";
        ctx!.lineWidth = 1.5;
        const lastIdx = knots.length - 1;
        const feedIdx = result.feed_knot_index;
        if (wi === feedWireIdx && feedIdx > 0 && feedIdx < lastIdx) {
          drawArmEnvelope(ctx!, knots, mags, magMaxGlobal, project, 0, feedIdx, envScale);
          drawArmEnvelope(ctx!, knots, mags, magMaxGlobal, project, feedIdx, lastIdx, envScale);
        } else {
          drawArmEnvelope(ctx!, knots, mags, magMaxGlobal, project, 0, lastIdx, envScale);
        }

        // Wire label near the leftmost knot for multi-wire geometries.
        if (result.wires.length > 1) {
          const lp = project(knots[0]);
          ctx!.fillStyle = "#7b8493";
          ctx!.font = "11px ui-monospace, monospace";
          ctx!.fillText(wire.label, lp.x - 8 - ctx!.measureText(wire.label).width, lp.y + 3);
        }
      }

      // Feed marker on the feed wire.
      const feedWire = result.wires[feedWireIdx];
      if (feedWire) {
        const feed = project(feedWire.knot_positions[result.feed_knot_index]);
        ctx!.fillStyle = "#ffd166";
        ctx!.beginPath();
        ctx!.arc(feed.x, feed.y, 5, 0, Math.PI * 2);
        ctx!.fill();
        ctx!.font = "12px ui-monospace, monospace";
        ctx!.fillText("feed", feed.x + 8, feed.y - 8);
      }

      // λ/4 scale bar.
      const barLenPx = (lambdaDesign / 4) * scale;
      const barX0 = pad;
      const barY = h - 24;
      ctx!.strokeStyle = "#7b8493";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(barX0, barY);
      ctx!.lineTo(barX0 + barLenPx, barY);
      ctx!.moveTo(barX0, barY - 4);
      ctx!.lineTo(barX0, barY + 4);
      ctx!.moveTo(barX0 + barLenPx, barY - 4);
      ctx!.lineTo(barX0 + barLenPx, barY + 4);
      ctx!.stroke();
      ctx!.fillStyle = "#9aa3b2";
      ctx!.font = "11px ui-monospace, monospace";
      ctx!.fillText(`λ/4 = ${(lambdaDesign / 4).toFixed(2)} m`, barX0, barY - 8);
    }

    onResize();
    const obs = new ResizeObserver(onResize);
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [result]);

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
  ctx.beginPath();
  for (let i = start; i <= end; i++) {
    const p = project(knots[i]);
    // Tangent from neighbors WITHIN this arm. At the apex (end of one arm,
    // start of the next) we look inward so each arm gets its own tangent.
    let dx = 0;
    let dy = -1;
    if (i < end) {
      const q = project(knots[i + 1]);
      dx = q.x - p.x;
      dy = q.y - p.y;
    } else if (i > start) {
      const q = project(knots[i - 1]);
      dx = p.x - q.x;
      dy = p.y - q.y;
    }
    const n = Math.hypot(dx, dy) || 1;
    let nx = -dy / n;
    let ny = dx / n;
    // Orient consistently toward screen-up so both arms' envelopes sit above
    // (outside) the V, mirroring each other.
    if (ny > 0) {
      nx = -nx;
      ny = -ny;
    }
    const offset = (mags[i] / magMax) * envScale;
    const ex = p.x + nx * offset;
    const ey = p.y + ny * offset;
    if (i === start) ctx.moveTo(ex, ey);
    else ctx.lineTo(ex, ey);
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
