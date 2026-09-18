# BucklingModeIdentification.jl

Modal identification of conventional finite strip buckling modes into **Global / Distortional / Local / Other** participations, on the section's **actual rounded-corner geometry**, from a centerline to an interactive viewer in a few calls.

The constrained finite strip method (cFSM, Ádány & Schafer) constrains the eigenproblem to one mechanical subspace and returns "pure" mode curves. This package does the complementary thing: it leaves the conventional (unconstrained) signature curve untouched and, at every half-wavelength, writes the actual buckled shape in cFSM's mechanical base,

    d = [R_G  R_D  R_L  R_O] c ,

so the participation of each class and the component shapes `d_M = R_M c_M` (which sum back to the mode) are available at every point of the curve — where the local trough is, where the distortional one is, and how much they interact.

## Quick start

```julia
using BucklingModeIdentification
using CairoMakie                      # optional: enables the static figures

corners = [(1.5, 0.5), (1.5, 0.0), (0.0, 0.0), (0.0, 2.5), (1.5, 2.5), (1.5, 2.0)]   # lipped Cee, in.
X, Y = polyline_with_fillets(corners, 0.1; n_arc = 4, n_flat = 6)                   # r = 0.1 in. fillets
t, E = 0.0451, 29500.0                                                               # in., ksi

axial = decompose_section(X, Y, t, E; name = "Cee-2.5x1.5", load = :axial)
summarize(axial)                      # local / distortional troughs with G-D-L-O %, pure-mode minima
write_outputs(axial; outdir = "out")  # figures, JSON, single-case interactive viewer

bending = decompose_section(X, Y, t, E; name = "Cee-2.5x1.5", load = :bending)
holed   = decompose_section(X, Y, t, E; name = "Cee-2.5x1.5", load = :axial, hole = 1.5)
build_viewer([axial, bending, holed]; outdir = "out")   # one page, profile + model menus
```

Any centerline works as input — your own coordinates (rounded corners as discretised arcs, ribs, jogs), one thickness or one per element via `cufsm_model`. Units are whatever you pass, consistently (length, force/length² for `E`); load factors come out in force (axial) or force·length (bending).

## What you get

- `SectionCase` with the `DecompositionResult`: half-wavelengths, load factors, participation matrix (G, D, L, O in %), the four component displacement fields, the constrained pure-L/pure-D curves, and the identified local (L-dominated) and distortional (D-dominated) troughs.
- Figures: signature curve with both troughs and the pure-mode minima marked, stacked participation, and the total shape split into its four components at each trough.
- Viewer: a self-contained HTML page (open from disk or publish) — scrub the half-wavelength, read the conventional and pure-mode values, see the shape components; `build_viewer` adds profile and model menus for many cases.

## Method notes

- **Rounded corners.** cFSM's criteria make every non-collinear node a corner, so a rounded mesh gets dozens of "corner-warping" D modes and misidentifies local buckling as distortional. This package builds the strictly membrane-strain-free (Vlasov) base on the real mesh, keeps one primary warping value per corner arc (its angular midpoint) plus the free edges, and statically condenses the other arc warpings through the section's own stiffness (`Kred = TᵀKT`) — each corner takes its minimum-energy shape (the idea of Beregszászi & Ádány's elastic corner elements). The condensed space orthogonal to G is D (same dimension as the sharp-corner topology); the leftover corner-arc bending directions are grouped with L. Nothing in the geometry is idealised. On a sharp-cornered mesh the construction is classic cFSM. `corner_model = :literal` gives the classic (unsuitable) behaviour for comparison.
- **Participation** is `‖d_M‖₂ / Σ‖d_k‖₂` over the component displacements — unique and independent of how each subspace is spanned. CUFSM's L1-of-coefficients is dimension-biased with this base and is not used.
- **Pure-mode curves** are constrained minima and therefore upper bounds of the conventional troughs; their level depends on how a corner is constrained. The conventional curve is the exact elastic answer; the participations classify it.
- **Holes** use the zero-thickness net-section strip (Moen & Schafer). The two halves are then mechanically disconnected: the local trough is meaningful, but beyond it "Other" carries the halves' relative motion and pure-D has no minimum.
- **Mesh hygiene.** `drop_short_segments` removes strips shorter than `0.1 t` (default): a strip a few percent of `t` long has bending stiffness ~10⁹× its neighbours and wrecks the conditioning. `CUFSM.strip` rounds node coordinates in place; the package builds its base first and hands `strip` copies.

## API

High level: `polyline_with_fillets`, `decompose_section`, `summarize`, `write_outputs`, `build_viewer`, `plotting_available`.
Model: `cufsm_model`, `zero_thickness_hole` (hole placed on the web, detected with `straight_runs` or given explicitly), `drop_short_segments`, `corner_topology`.
Core: `decompose_modes`, `pure_mode_curve`, `characteristic_minima`, `discretized_shape`.
Figures (with a Makie backend loaded): `plot_decomposition`, `plot_mode_components`.
Viewer files: `export_viewer_json`, `write_viewer_html`, `write_profile_js`, `write_multi_viewer_html`.

## Installation

Depends on the RunToSolve packages `CUFSM`, `cFSM` and `CUFSMModalGeometry` (registered in `RunToSolveJuliaRegistry`; the `[sources]` in `Project.toml` also point at their GitHub repositories).

```julia
] registry add https://github.com/runtosolve/RunToSolveJuliaRegistry
] add BucklingModeIdentification        # once registered; until then: ] dev path/to/BucklingModeIdentification.jl
```

## References

- S. Ádány, B. W. Schafer, "Buckling mode decomposition of single-branched open cross-section members via finite strip method: derivation / application and examples", *Thin-Walled Structures* 44 (2006).
- S. Ádány, B. W. Schafer, "A full modal decomposition of thin-walled, single-branched open cross-section members via the constrained finite strip method", *J. Constructional Steel Research* 64 (2008).
- Z. Beregszászi, S. Ádány, "Modal buckling analysis of thin-walled members with rounded corners by using the constrained finite strip method with elastic corner elements", *Thin-Walled Structures* 142 (2019).
- Z. Li, B. W. Schafer, "Buckling analysis of cold-formed steel members with general boundary conditions using CUFSM: conventional and constrained finite strip methods", CCFSS (2010).
