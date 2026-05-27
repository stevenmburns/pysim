import { useEffect, useRef, useState } from "react";

type SolveResponse = {
  knot_positions: [number, number, number][];
  knot_currents_re: number[];
  knot_currents_im: number[];
  feed_knot_index: number;
  z_in_re: number;
  z_in_im: number;
  angle_deg: number;
  n_per_arm: number;
  design_freq_mhz: number;
  measurement_freq_mhz: number;
  halfdriver_factor: number;
  arm_len_m: number;
  solve_ms: number;
};

type SolveRequest = {
  angle_deg: number;
  n_per_arm: number;
  design_freq_mhz: number;
  measurement_freq_mhz: number;
  halfdriver_factor: number;
  wire_radius: number;
};

const WS_URL = `ws://${window.location.host}/ws`;

export function App() {
  const [angle, setAngle] = useState(30);
  const [nPerArm, setNPerArm] = useState(30);
  const [designFreq, setDesignFreq] = useState(13.625);
  const [measFreq, setMeasFreq] = useState(13.625);
  const [halfdriverFactor, setHalfdriverFactor] = useState(0.962);
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

  const wsRef = useRef<WebSocket | null>(null);
  const inFlightRef = useRef(false);
  const pendingRef = useRef<SolveRequest | null>(null);
  const sendStartRef = useRef(0);

  // The latest control values, used to send a new request when the prior one
  // completes (drops intermediate values rather than queuing them all up).
  const controlsRef = useRef<SolveRequest>({
    angle_deg: angle,
    n_per_arm: nPerArm,
    design_freq_mhz: designFreq,
    measurement_freq_mhz: measFreq,
    halfdriver_factor: halfdriverFactor,
    wire_radius: wireRadius,
  });

  useEffect(() => {
    controlsRef.current = {
      angle_deg: angle,
      n_per_arm: nPerArm,
      design_freq_mhz: designFreq,
      measurement_freq_mhz: measFreq,
      halfdriver_factor: halfdriverFactor,
      wire_radius: wireRadius,
    };
    requestSolve();
  }, [angle, nPerArm, designFreq, measFreq, halfdriverFactor, wireRadius]);

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
        <h1>inverted V — interactive</h1>

        <div className="group-label">antenna</div>

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

        <div className="field">
          <label>
            <span>design freq</span>
            <span>{designFreq.toFixed(3)} MHz</span>
          </label>
          <input
            type="range"
            min={1}
            max={30}
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
            <span>segments / arm (N)</span>
            <span>{nPerArm}</span>
          </label>
          <input
            type="range"
            min={10}
            max={80}
            step={1}
            value={nPerArm}
            onInput={(e) => setNPerArm(Number((e.target as HTMLInputElement).value))}
          />
        </div>

        <div className="field">
          <label>
            <span>measurement freq</span>
            <span>{measFreq.toFixed(3)} MHz</span>
          </label>
          <input
            type="range"
            min={1}
            max={30}
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
          <div className="row">
            <span>arm length</span>
            <span className="val">{result ? `${result.arm_len_m.toFixed(3)} m` : "—"}</span>
          </div>
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
            <span>rtt</span>
            <span className="val">{rttMs != null ? `${rttMs.toFixed(1)} ms` : "—"}</span>
          </div>
        </div>
      </aside>

      <main className="stage">
        <CurrentCanvas result={result} />
        <div className="status">ws: {status}</div>
      </main>
    </div>
  );
}

function feedMag(r: SolveResponse): number {
  const re = r.knot_currents_re[r.feed_knot_index];
  const im = r.knot_currents_im[r.feed_knot_index];
  return Math.hypot(re, im);
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

      // Draw axes guide (apex at top-center).
      ctx!.strokeStyle = "#23272f";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(w / 2, 20);
      ctx!.lineTo(w / 2, h - 20);
      ctx!.stroke();

      if (!result) return;

      // Project knots from (x, z) onto the canvas. Scale is anchored to the
      // physical arm length so the perceived arm length doesn't change as the
      // user sweeps the droop angle. Apex stays at a fixed screen position
      // and the arms swing around it.
      const knots = result.knot_positions;
      const apex = knots[result.feed_knot_index];
      const armLen = Math.hypot(
        knots[0][0] - apex[0],
        knots[0][1] - apex[1],
        knots[0][2] - apex[2],
      ) || 1e-9;
      const pad = 50;
      // Sized so that a flat dipole (2*armLen wide) fits horizontally and a
      // fully-drooped V (armLen tall) fits vertically.
      const scale = Math.min((w - 2 * pad) / 2, h - 2 * pad) / armLen;

      const project = (p: [number, number, number]) => {
        const cx = w / 2;
        const apexY = pad + 30;
        return {
          x: cx + (p[0] - apex[0]) * scale,
          y: apexY + (apex[2] - p[2]) * scale, // larger droop hangs further down
        };
      };

      // Current magnitude per knot.
      const re = result.knot_currents_re;
      const im = result.knot_currents_im;
      const mags = re.map((r, i) => Math.hypot(r, im[i]));
      const magMax = Math.max(...mags, 1e-30);

      // Wire trace.
      ctx!.lineCap = "round";
      ctx!.lineJoin = "round";
      for (let i = 0; i < knots.length - 1; i++) {
        const a = project(knots[i]);
        const b = project(knots[i + 1]);
        const m = 0.5 * (mags[i] + mags[i + 1]) / magMax;
        ctx!.strokeStyle = currentColor(m);
        ctx!.lineWidth = 2 + 6 * m;
        ctx!.beginPath();
        ctx!.moveTo(a.x, a.y);
        ctx!.lineTo(b.x, b.y);
        ctx!.stroke();
      }

      // Feed marker.
      const feed = project(knots[result.feed_knot_index]);
      ctx!.fillStyle = "#ffd166";
      ctx!.beginPath();
      ctx!.arc(feed.x, feed.y, 5, 0, Math.PI * 2);
      ctx!.fill();
      ctx!.fillStyle = "#ffd166";
      ctx!.font = "12px ui-monospace, monospace";
      ctx!.fillText("feed", feed.x + 8, feed.y - 8);

      // Current magnitude curve along the wire, drawn as a "skyline" offset
      // perpendicular to the wire. Each arm gets its own stroke: the
      // perpendicular direction flips at the apex, so a single continuous
      // envelope would bow-tie across the feed.
      ctx!.strokeStyle = "rgba(118, 208, 255, 0.7)";
      ctx!.lineWidth = 1.5;
      const envScale = 60;
      const feedIdx = result.feed_knot_index;
      drawArmEnvelope(ctx!, knots, mags, magMax, project, 0, feedIdx, envScale);
      drawArmEnvelope(ctx!, knots, mags, magMax, project, feedIdx, knots.length - 1, envScale);
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
