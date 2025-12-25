Current implemented model (code-accurate)

Universe / lattice

* Spacetime is a 2D square lattice with periodic boundaries (implemented via `np.roll`).
* The primary matter field is a complex scalar field ψ(y,x,t) stored as a complex NumPy array.
* The default local coupling operator is an isotropic 2D stencil using axial + diagonal neighbors:
	∇²_iso f = (4/6)(axial − 4f) + (1/6)(diagonal − 4f).
	This improves isotropy but is not continuum-normalized (it rescales the effective dispersion).

Matter dynamics (ψ)

* ψ does NOT currently evolve via a second-order-in-time Klein–Gordon update.
* The implemented microscopic law is first-order in time (a Schrödinger-like “phase dynamics” in local proper time τ):
	i dψ/dτ = Hψ + drive,
	where Hψ = ω ψ + κ ∇²_iso ψ.
* Time integration is explicit and time-reversible via a leapfrog scheme split across Re/Im parts.
	The integrator stores Re(ψ) on integer steps and Im(ψ) on half-steps.

Driving / sources

* Two source patterns are implemented:
	- Monochromatic complex driving at fixed lattice sites: drive ∝ exp(-i Ω t)
	- A Gaussian pulse drive in time at a site.
* Driving is essential in practice for stable, clean interference patterns in finite lattices.

Time dilation (“lapse”) and optional backreaction

* A scalar clock-rate field N(y,x,t) can be used as a local lapse via dτ = dt · N.
* Two modes exist:
	- Prescribed / backreacting lapse: N is computed from ψ via a local curvature proxy C(ψ) and a monotone map (exp or rational), with clipping.
	- Dynamical lapse field (“gravity”): N itself evolves via a second-order wave equation
		N_tt = c_g^2 ∇²_iso N + source(ψ) − μ^2 (N − base) − γ N_t,
		with local finite-speed propagation, and ψ is stepped using midpoint coupling to N.