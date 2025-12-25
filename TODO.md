Spacetime / gravity — reorganized actionable plan

Goal: break the high-level limitations into independent, implementable tasks ordered so each can be completed without being blocked by others. Tasks are grouped by dependency level: Immediate (no prerequisites), Short-term (small dependencies), Medium (require completed short-term items), and Long-term (large architectural work).

Immediate — can be implemented and tested now
- Add and harden unit tests and small utilities:
	- Expand tests for `curvature_proxy`, `clock_rate_from_psi`, `laplacian_iso`, and leapfrog init/step to validate behavior on small grids (files: `code/model.py`, `code/measurements.py`, `tests/`), and include NaN/Inf checks.
	- Add deterministic RNG seeds and smaller grid smoke-tests to make CI fast and reliable.
	- Improve measurement helpers to expose stable intermediate diagnostics (e.g., radial cache invariants).
	Why: low risk, directly increases confidence and enables later changes.

Short-term — small dependencies, self-contained
- Robustify clock-rate (lapse) code path:
	- Centralize and test clock-rate clipping, broadcasting, and per-site dt scaling (no global side effects).
	- Add unit tests that exercise `clock_rate_from_psi` and `clock_rate_from_curvature` with edge cases and clipping.
	Why: keeps later physics changes from being blocked by numerical surprises.

- Improve energy/hamiltonian diagnostics and conservation tests:
	- Add a clear API to compute KE/PE and weighted norms (`hamiltonian_total`, `energy_density`, `weighted_norm`), and unit tests comparing conserved quantities in controlled scenarios.
	Why: Allows independent verification of later physical couplings.

Medium — higher-level physics features with light prerequisites
- Make sourcing less heuristic (incremental):
	- Implement a small, testable abstraction `gravity_source_from_fields(...)` that accepts different source kinds (density, curvature) and returns normalized, clipped sources. Keep existing proxy computations behind this API so callers don't rely on the heuristic directly.
	- Add unit tests comparing different source kinds on frozen fields.
	Why: This decouples callers from the exact choice of source mapping and enables swapping/repair later.

- Local lapse dynamics (wave/field evolution) improvements:
	- Harden `init_clock_rate_wave`, `step_clock_rate_wave` with clear boundary/clip behavior and unit tests for causality on small domains.
	- Add regression tests that validate that the probe at known distances remains unchanged until the wavefront arrives.
	Why: This is required before coupling clock evolution to matter; can be developed after the clock-rate hardening.

Long-term — large architectural or conceptual changes (may require project redesign)
- Replace heuristic sourcing with (approximate) conserved stress-energy coupling:
	- Design a discrete stress-energy density evaluator (local, testable) and a minimal conservation-aware coupling; start by implementing local energy density sampling and unit tests.
	- Make this a staged improvement: first read-only diagnostics, then a one-way coupling (matter -> gravity) before attempting full two-way conserved evolution.
	Why: This is a large change — split into small, testable steps to avoid blocking.

- Add additional degrees of freedom (vector fields, multiple species):
	- Introduce a clear plugin pattern for adding fields (scalar/vector) and per-species parameters (mass, charge). Implement one extra scalar species as a proof of concept.
	Why: Enables experiment without forcing a full rewrite.

- Symmetry/continuum and performance
	- Add targeted tests for isotropy, scaling, and dispersion on finer meshes; expose options for improved stencils.
	- Profile and isolate hotspots; add optional faster NumPy/Numba paths behind clear, covered APIs.
	Why: These are optional optimizations and require stable APIs/tests first.

Research / aspirational (conceptual — do after smaller steps)
- Lorentz invariance, local inertial frames, and equivalence principle enforcement — plan experimental designs, but do not block core progress on these.
- Measurement theory / quantization / fermions / gauge fields — separate research tracks; implement small, self-contained prototypes if desired.

Implementation notes
- Make each task self-contained: add targeted unit tests first, then implementation. Keep changes small and reviewable.
- For any task that changes numerics, add regression tests that compare behavior on small deterministic inputs before enabling larger runs.
- Prioritize building small APIs (adapters/wrappers) rather than wholesale replacement; this avoids blocking dependent work.

Next steps
- I can (a) translate the Immediate and Short-term items into a sequence of concrete PR-sized tasks and create issue-style todos, or (b) start implementing the first Immediate task (tests and measurement helpers). Which would you prefer?

----
Original notes (kept for reference): list of high-level limitations from the earlier file — preserved, but reorganized above into implementable tasks.