"""P0 verification for the element-aware array-block solver.

Confirms the load-bearing measurements behind `docs/array_block_solver_plan.md`
before the solver is built on top of them:

  1. Element grouping is clean (equal-size elements for an identical-element
     array).
  2. Self-blocks are identical across elements (free space) — the property the
     single-shared-self-block reuse relies on.
  3. Coupling blocks are weak: mean ‖Z_ab‖_F / ‖Z_aa‖_F is a few percent.
  4. Coupling blocks are low rank: numerical rank ~4 at a 1% threshold.

Also reports the storage an element-aware decomposition would reach
(self-fraction + low-rank coupling) vs the generic H-matrix's compression.

Run:  PYTHONPATH=. .venv/bin/python scripts/array_block_verify.py
"""

import numpy as np

from antenna_designer.engines.pysim import PysimEngine
from antenna_designer.designs import bowtiearray2x4, invveearray

from pysim.hmatrix import HMatrixPySim
from pysim.array_block import element_groups


def _build_sim(mod, solver, degree=2):
    builder = mod.Builder()
    eng = PysimEngine(builder, solver=solver, solver_kwargs={"degree": degree})
    wl = eng._wavelength_for(builder.freq)
    return eng._make_solver(wavelength=wl), builder


def _numerical_rank(block, rel_tol=0.01):
    """Numerical rank at a relative singular-value threshold."""
    s = np.linalg.svd(block, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 0
    return int(np.count_nonzero(s > rel_tol * s[0]))


def verify(name, mod, rank_tol=0.01):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    sim, builder = _build_sim(mod, HMatrixPySim)
    part = element_groups(sim)
    n = part.n_basis
    P = part.n_elem
    print(f"n_basis = {n}   n_elem = {P}   element sizes = {part.sizes.tolist()}")

    equal_size = len(set(part.sizes.tolist())) == 1
    print(
        f"  [grouping] equal-size elements: {equal_size}   "
        f"n_shapes={part.n_shapes}   shape_of_elem={part.shape_of_elem.tolist()}"
    )

    # ---- self-blocks: identical *within each geometric shape class* ---------
    # An array generally has several distinct element shapes (up to 4 for a
    # 2x4: itop/ibot/otop/obot), so self-blocks match only within a shape
    # class. Free space + consistent intra-element ordering ⇒ ~1e-12 there.
    self_blocks = [sim.zblock(g, g) for g in part.groups]
    worst_self = 0.0
    for s in range(part.n_shapes):
        members = [e for e in range(P) if part.shape_of_elem[e] == s]
        ref = self_blocks[members[0]]
        rnorm = np.linalg.norm(ref)
        for e in members[1:]:
            worst_self = max(worst_self, np.linalg.norm(self_blocks[e] - ref) / rnorm)
    print(
        f"  [self-blocks] max ‖Z_aa - Z_a'a'‖_F / ‖Z‖_F within shape class: "
        f"{worst_self:.2e}"
    )

    # ---- coupling blocks ---------------------------------------------------
    ratios = []
    ranks = []
    for a in range(P):
        for b in range(P):
            if a == b:
                continue
            Cab = sim.zblock(part.groups[a], part.groups[b])
            ratios.append(np.linalg.norm(Cab) / np.linalg.norm(self_blocks[a]))
            ranks.append(_numerical_rank(Cab, rel_tol=rank_tol))
    ratios = np.array(ratios)
    ranks = np.array(ranks)
    # Coupling measured with exact connectivity grouping; far weaker than the
    # plan's k-means estimate of ~0.02 (k-means mis-assigned boundary bases,
    # leaking strong near-field self terms into the "coupling" blocks).
    print(
        f"  [coupling] ‖Z_ab‖_F/‖Z_aa‖_F  mean={ratios.mean():.2e}  "
        f"max={ratios.max():.2e}"
    )
    print(
        f"  [coupling] numerical rank @ {rank_tol:.0%}  "
        f"mean={ranks.mean():.1f}  max={ranks.max()}  "
        f"(block ~{part.sizes[0]}x{part.sizes[0]})"
    )

    # ---- storage estimate --------------------------------------------------
    Nself = int(part.sizes[0])
    # P dense self-blocks (operator) vs n_shapes distinct ones (factor reuse),
    # plus P(P-1) coupling blocks at the measured mean rank.
    rmean = ranks.mean()
    self_store_all = P * Nself * Nself
    self_store_shared = part.n_shapes * Nself * Nself
    coup_store = P * (P - 1) * rmean * (2 * Nself)
    dense = n * n
    print(
        f"  [storage] dense=n²={dense}  "
        f"self(all P)+coupling={(self_store_all + coup_store) / dense:.1%}  "
        f"self({part.n_shapes} shapes, factor-reuse)+coupling="
        f"{(self_store_shared + coup_store) / dense:.1%}"
    )

    return {
        "name": name,
        "n": n,
        "P": P,
        "equal_size": equal_size,
        "worst_self": worst_self,
        "coupling_mean": float(ratios.mean()),
        "coupling_max": float(ratios.max()),
        "rank_mean": float(ranks.mean()),
        "rank_max": int(ranks.max()),
    }


def main():
    results = []
    for name, mod in [
        ("bowtiearray2x4", bowtiearray2x4),
        ("invveearray", invveearray),
    ]:
        results.append(verify(name, mod))

    print(f"\n{'=' * 70}\nP0 gate summary\n{'=' * 70}")
    for r in results:
        self_ok = r["worst_self"] is not None and r["worst_self"] < 1e-9
        weak_ok = r["coupling_mean"] < 0.10
        lowrank_ok = r["rank_max"] <= 8
        verdict = "PASS" if (self_ok and weak_ok and lowrank_ok) else "CHECK"
        print(
            f"  {r['name']:<16} self={r['worst_self']:.1e} "
            f"coupling_mean={r['coupling_mean']:.2e} rank_max={r['rank_max']} "
            f"-> {verdict}"
        )


if __name__ == "__main__":
    main()
