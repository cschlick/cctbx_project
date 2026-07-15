# RSR_MAP — Core Real-Space Refinement Code Path

A map of the **core** real-space refinement (RSR) code path in cctbx/mmtbx: how a set of
atomic coordinates is refined against a real-space density map. Focus is on the
gradient-driven "individual sites" path (LBFGS coordinate refinement), from the top-level
mmtbx driver down to the C++ interpolation kernels.

Scope note: this maps the *optimization* path (target + gradients + LBFGS). It does **not**
cover the combinatorial/torsion sampling fitters (`fit_residue*.py`, `fit_residues*.py`,
`explode_and_refine.py`), which are a separate family that ultimately score candidates using
the same interpolation kernels described in §1.

---

## 0. The one-paragraph summary

The heart of RSR is a **map-interpolation kernel** in
`cctbx/maptbx/interpolation.h`. Given a continuous atomic position, it looks up the density
value (and, for the analytic variants, the spatial gradient of the density) at that position
by interpolating the discretely-sampled map. The RSR **target** is the (negated) sum of
interpolated density values over the atoms; the **gradient** is the sum of the density
gradients. Those two numbers are handed to **LBFGS**, which moves the atoms uphill in
density. Geometry restraints (bonds, angles, …) are added to the same target/gradient so the
model stays chemically sensible. Everything above the C++ kernel is orchestration: weight
determination, selections, LBFGS bookkeeping, and NCS handling.

---

## 1. The interpolation kernels — `cctbx/maptbx/interpolation.h`

This is the file you were thinking of: several methods for interpolating a map onto sites,
in increasing sophistication. All are templated free functions in
`namespace cctbx::maptbx`.

Shared helper: **`get_corner`** (`interpolation.h:14`) — converts a fractional (or Cartesian,
via a gridding matrix) coordinate into (a) the integer grid index of the lower corner of the
enclosing voxel and (b) per-axis fractional weights `weights_[axis][0/1]`. Three constructors
handle the periodic/crystallographic case, the non-crystallographic (gridding-matrix) case,
and the ASU-mapped case.

### Value-only interpolants (target only, no analytic gradient)

| Function | Line | Stencil | Notes |
|---|---|---|---|
| `eight_point_interpolation` | 250 / 271 / 292 | 2×2×2 (trilinear) | Overloaded for `flex_grid`, `c_grid_padded<3>`, `c_grid<3>`. The workhorse value lookup. |
| `tricubic_interpolation` | 453 / 494 | 4×4×4 | Uses `scitbx::math::interpolate_at_point`; overloaded for padded and non-padded grids. |
| `non_crystallographic_eight_point_interpolation` | 546 | 2×2×2 | Trilinear with a gridding matrix + out-of-bounds handling (no periodic wrap). |
| `asu_eight_point_interpolation` | 581 | 2×2×2 | Trilinear with ASU back-mapping for grid points on the ASU's open face. |

### Value + analytic-gradient interpolants (the ones used for gradient refinement)

These return `af::tiny<MapFloatType, 4>` = `(value, d/dx, d/dy, d/dz)`, gradients divided by
the per-axis `step` (grid spacing) to give a Cartesian-ish derivative.

| Function | Line | Stencil | "Sophistication" |
|---|---|---|---|
| `eight_point_interpolation_with_gradients` | 103 | 2×2×2 | **linear** — trilinear value, gradient from the trilinear form. Cheapest. |
| `quadratic_interpolation_with_gradients` | 159 | needs ±1 / +2 neighbors | **quadratic** — fits a local quadratic; branches on whether the fractional coord is in the lower/upper half of the voxel to pick the one-sided stencil. |
| `tricubic_interpolation_with_gradients` | 400 | 4×4×4 | **tricubic** — the most sophisticated. Builds a 4×4×4 stencil `f[4][4][4]`, evaluates a separable cubic (`cubic`, line 311) and its derivative (`gcubic`, line 327) along each axis via the `fxpq/fqyp/fpqz/fxyq/fqyz/fxqz` helper ladder (lines 341–394). Smoothest value and continuous gradient. |

The `cubic`/`gcubic` pair encodes the actual 1-D cubic polynomial coefficients (with an
alternate coefficient set left in comments). This is where the "most sophisticated
interpolation" lives.

---

## 2. The target-and-gradients driver — `cctbx/maptbx/target_and_gradients.h`

Wraps the kernels into target/gradient objects. Three namespaces:

- **`target_and_gradients::diffmap::compute`** (`target_and_gradients.h:11`) — difference-map
  target: builds a `-2*(map_target - map_current)` difference array, then finite-difference
  gradients via `eight_point_interpolation`. Used by the diff-map refinement path.
- **`target_and_gradients::simple::compute`** (`target_and_gradients.h:269`) — **the core RSR
  target/gradient object.** Two constructors:
  - **FD constructor** (line 273): takes a `delta`, computes gradients by **finite
    differences** of `eight_point_interpolation` (±delta along each axis). This is the
    `"fd"` path.
  - **Analytic constructor** (line 306): takes an `interpolation` string and dispatches to
    the `_with_gradients` kernels:
    - `"linear"` → `eight_point_interpolation_with_gradients`
    - `"quadratic"` → `quadratic_interpolation_with_gradients`
    - `"tricubic"` → `tricubic_interpolation_with_gradients`
    - anything else → `throw "Unknown interpolation mode."`
- **`target_and_gradients::simple::target(...)`** overloads (lines 460–601) — value-only
  targets and `target_per_site`, plus `magnification*` (map-scaling refinement, uses
  tricubic) and `score` (binary in/out-of-density scoring). Not on the gradient hot path.

---

## 3. Python bindings — `cctbx/maptbx/boost_python/maptbx_ext.cpp`

Exposes the C++ to Python (module `cctbx.maptbx`, i.e. `maptbx.ext`):

- **`target_and_gradients_simple`** class (`maptbx_ext.cpp:183`) — binds
  `simple::compute<double>` with **two** `init` overloads matching the two C++ constructors:
  - `(unit_cell, map_target, sites_cart, delta, selection)` → FD path (line 184)
  - `(unit_cell, map_target, sites_cart, selection, interpolation)` → analytic path (line 194)
  - `.target()` and `.gradients()` accessors.
- `real_space_target_simple` (several overloads, line 209+) — value-only targets.
- `target_and_gradients_simple_magnification` (line 138) — map magnification.
- Direct kernel bindings (line 1184+): `eight_point_interpolation`,
  `eight_point_interpolation_with_gradients`, `quadratic_interpolation_with_gradients`,
  `tricubic_interpolation[_with_gradients]`, `non_crystallographic_eight_point_interpolation`,
  `asu_eight_point_interpolation`.

---

## 4. Python LBFGS core — `cctbx/maptbx/real_space_refinement_simple.py`

The class **`lbfgs`** (`real_space_refinement_simple.py:107`) is the minimizer that ties the
target object to `scitbx.lbfgs`.

- `__init__` sets up selections, the map, geometry restraints, weight, and the LBFGS run
  (line 168, `scitbx.lbfgs.run(target_evaluator=O, ...)`).
- **`compute_functional_and_gradients`** (line 179) is the LBFGS callback and the real hot
  loop. Per evaluation it:
  1. Reshapes the flat parameter vector into `sites_cart_variable` (with `damp_shifts` to cap
     per-step motion, line 189).
  2. Computes the real-space term by branching on `gradients_method` (line 199):
     - `"fd"` → `maptbx.target_and_gradients_simple(..., delta=..., selection=...)` (FD ctor)
     - else (`"linear"/"quadratic"/"tricubic"`) →
       `maptbx.target_and_gradients_simple(..., selection=..., interpolation=gradients_method)`
       (analytic ctor)
     - (or the `local_standard_deviations_*` variant when a smoothing radius is set)
  3. Negates and weights: `rs_f *= -weight; rs_g *= -weight` (density is *maximized*, LBFGS
     *minimizes*).
  4. Adds geometry restraints via `geometry_restraints_manager.energies_sites(...)` (line 243)
     so `f = rs_f + gr_e.target`, `g = rs_g + gr_e.gradients`.

Other pieces in this file: `local_standard_deviations_*` (a normalized target with FD
gradients, used when `local_standard_deviations_radius` is set) and
`magnification_anisotropic_minimization` (map-scale refinement, off the main path).

---

## 5. mmtbx driver layer — `mmtbx/refinement/real_space/individual_sites.py`

The public entry points most callers use.

- **`easy`** (`individual_sites.py:14`) — "simplest interface to most automated and fast
  real-space refinement." Given `map_data`, `xray_structure`, `pdb_hierarchy`,
  `geometry_restraints_manager`:
  1. Determines the RSR-vs-geometry **weight** `w` via
     `mmtbx.refinement.real_space.weight.run(...)` (line 44) if not supplied.
  2. Constructs a `simple` refiner (line 55) and calls `.refine(weight, xray_structure)`.
  3. Writes refined sites back into the `xray_structure` / `pdb_hierarchy`.
  - `gradients_method` is validated here: `assert gradients_method in
    ["fd", "linear", "quadratic", "tricubic"]` (line 34).
- **`simple`** (`individual_sites.py:70`) — sets LBFGS parameters (`stpmax=25`, max
  iterations, exception handling) and, in `.refine()` (line 96), calls
  `maptbx.real_space_refinement_simple.lbfgs(...)` (line 100) — i.e. §4 — passing
  `gradients_method` straight through. When `ncs_groups` is set it instead routes through
  `mmtbx.refinement.minimization_ncs_constraints` (line 116).
- Further down the file (line ~306) is a lower-level iterative driver taking the same
  `gradients_method="fd"` default.

Weight determination (`mmtbx/refinement/real_space/weight.py`) itself runs trial
refinements, so it also carries `gradients_method` (default `"fd"`, line 14) to stay
consistent with the real run.

---

## 6. End-to-end call chain (gradient path)

```
mmtbx.refinement.real_space.individual_sites.easy            # user entry
  └─ weight.run(...)                                         # pick RSR/geometry weight
  └─ individual_sites.simple.refine(weight, xray_structure)
       └─ cctbx.maptbx.real_space_refinement_simple.lbfgs    # §4  Python LBFGS
            └─ scitbx.lbfgs.run(target_evaluator=self)
                 └─ lbfgs.compute_functional_and_gradients   # per-iteration callback
                      ├─ maptbx.target_and_gradients_simple  # §3 binding
                      │    └─ simple::compute<double>         # §2  C++
                      │         ├─ (fd)  eight_point_interpolation ±delta      # §1 trilinear
                      │         └─ (analytic) {eight_point|quadratic|tricubic} # §1 kernels
                      │              _interpolation_with_gradients
                      └─ geometry_restraints_manager.energies_sites(...)       # + restraints
```

---

## 7. Important nuance: what is actually the DEFAULT?

Worth flagging, since it's a common misconception (and slightly different from "the most
sophisticated method is the default"):

- The **default `gradients_method` is `"fd"`** (finite differences) at every caller in the
  tree — `easy`, `simple`, the lower-level driver, and `weight.run` all default to `"fd"`
  (`individual_sites.py:25,76,306`; `weight.py:14`). No PHIL layer overrides this to
  something else.
- Under `"fd"`, the value lookups use **`eight_point_interpolation`** (trilinear), and
  gradients are finite differences of that. So the **de facto default interpolation is
  trilinear**, not tricubic.
- **`tricubic` is the most sophisticated interpolant available** and is the smoothest
  analytic-gradient option, but it is **opt-in** — a caller must pass
  `gradients_method="tricubic"`. It is the default only inside the `magnification` map-scaling
  target (`target_and_gradients.h:104`), not in coordinate RSR.

So: tricubic is the "final, most sophisticated" method in the file, but the shipped
**cctbx-level** default for actual real-space refinement is finite-difference gradients over
trilinear interpolation.
If you intend to work per-atom on the analytic-gradient path (the `rsr_peratom` branch name
suggests per-atom behavior), the code to touch is the analytic constructor of
`simple::compute` (`target_and_gradients.h:306`) and the `_with_gradients` kernels in
`interpolation.h`.

### 7a. But Phenix (the primary consumer) defaults to TRICUBIC

The cctbx defaults above are the *library* defaults. The **Phenix `phenix.real_space_refine`
program overrides them to `tricubic`** at the PHIL layer:

- `phenix/refinement/rsr/master.params:66` —
  `gradients_method = fd linear quadratic *tricubic` (the `*` = PHIL default → **tricubic**).
  Not overridden by the adjacent `customization.params` overlay.
- Loaded by `phenix/programs/real_space_refine.py:41`
  (`master_phil_str = rsr.master_params_str`, via `phenix/refinement/rsr/parameters.py`).
- `phenix/refinement/macro_cycle_real_space.py:364` passes `self.params.gradients_method`
  into `mmtbx...individual_sites.simple(...)`, which flows to the cctbx `lbfgs` analytic
  branch (§4) → `tricubic_interpolation_with_gradients`.

Caveats:
- It's a user-exposed `choice`, so `gradients_method=fd|linear|quadratic` overrides it per run.
- Not every Phenix path uses tricubic: `phenix/utilities/rs_refine.py:188` hardcodes `"fd"`.
- The legacy GUI PHIL `wxGUI2/Programs/Refine/real_space_refine_pre_1_19.py:81` also shows
  `*tricubic`, i.e. this default predates the 1.19 refactor.

Net: for the real-world "phenix.real_space_refine" path, **tricubic is the effective
default**; `"fd"` is only the bare-cctbx default.

---

## 8. File index (core path)

| Layer | File | Key symbols |
|---|---|---|
| Interpolation kernels | `cctbx/maptbx/interpolation.h` | `get_corner`, `eight_point_interpolation[_with_gradients]`, `quadratic_interpolation_with_gradients`, `tricubic_interpolation[_with_gradients]`, `cubic`/`gcubic` |
| Target/gradient objects | `cctbx/maptbx/target_and_gradients.h` | `simple::compute` (FD + analytic ctors), `simple::target`, `diffmap::compute`, `magnification` |
| Python bindings | `cctbx/maptbx/boost_python/maptbx_ext.cpp` | `target_and_gradients_simple`, `real_space_target_simple`, kernel `def(...)`s |
| Python LBFGS | `cctbx/maptbx/real_space_refinement_simple.py` | `lbfgs`, `compute_functional_and_gradients` |
| mmtbx driver | `mmtbx/refinement/real_space/individual_sites.py` | `easy`, `simple` |
| Weight determination | `mmtbx/refinement/real_space/weight.py` | `run` |
