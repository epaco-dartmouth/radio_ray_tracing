"""
Pulsar Magnetosphere O<->X Mode-Conversion Ray Tracer
======================================================

Hamiltonian ray tracing of radio waves through a pulsar magnetosphere:
cold, strongly-magnetized dispersion relation in a dipolar field with a
Goldreich-Julian-like density profile. O-modes refract self-consistently
in the plasma density gradient; X-modes travel in straight lines.

O<->X mode conversion is a single, path-dependent stochastic process: at
every integration step a local conversion probability is drawn from the
non-adiabatic Landau-Zener formula

    eta_{O/A->X} = sqrt(2) * p * (1-p)^alpha,   alpha = 0.6
    p = exp(-Delta)
    Delta = 2 * ktilde_x^3 * rhotilde * sin^3(phi) / (3 |eps^2 - 1|)

evaluated from the LOCAL field-line curvature frame at the ray's current
position (no persistent per-ray frame, no discrete "crossing" event to
detect -- eta is a smooth function of position). The coherence length for
one passage is tilde_k_x * rho (not rho alone -- rho is just the field-line
curvature scale from theta=xi/rho; the coupling is significant over
xi ~ tilde_k_x * rho, confirmed numerically against a traced near-peak
window), so the local rate is eta/(tilde_k_x*rho) and the per-step
probability is the Poisson form 1-exp(-rate*ds), which converges to
integral(rate ds) along the ray as the step size shrinks.

Units: lengths in stellar radii R_star, time in R_star/c, wavevectors k in
units of omega/c (so |k| equals the local refractive index n). The wave
frequency omega is a SINGLE FIXED number for an entire run -- use
model_from_physical() to derive all frequency-dependent Model fields
consistently from physical neutron-star parameters, rather than hand-editing
them separately.

Usage on a cluster:
    python pulsar_ox_raytracer.py --N 1000000 --r-em 1.5 --f-O 0.9 \\
        --cap-deg 20.0 --h-frac 0.002 --seed 1 --out res_out.h5
    python pulsar_ox_raytracer.py --help
"""

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import h5py
import numpy as np

rng = np.random.default_rng(42)


# =====================================================================
# Dipole field and plasma parameters
# =====================================================================

def Bfield(x, m):
    """Dipole magnetic field, normalized to |B|=1 at the magnetic pole on
    the stellar surface.
    Args:
        x : (N,3) float array -- Cartesian positions in R_star.
        m : Model             -- uses m.mhat.
    Returns:
        (N,3) float array -- B in code units.
    """
    r    = np.linalg.norm(x, axis=-1, keepdims=True)
    rhat = x / r
    mdr  = rhat @ m.mhat
    return 0.5 * (3.0 * mdr[..., None] * rhat - m.mhat) / r**3


def Bmag(x, m):
    """|B| at each position (1 at the polar surface)."""
    return np.linalg.norm(Bfield(x, m), axis=-1)


def wp2_w2(x, m):
    """(omega_p/omega)^2, GJ-like density n_e ~ |B| ~ 1/r^3."""
    return m.wp2_surf * Bmag(x, m)


def wB_w(x, m):
    """omega_B/omega at each position, ~ |B| ~ 1/r^3."""
    return m.wB_surf * Bmag(x, m)


# =====================================================================
# Dispersion relations
# =====================================================================

def H_O_isotropic(x, k, m):
    """O-mode, unmagnetized limit: H = 0.5*(|k|^2 + wp2 - 1)."""
    return 0.5 * (np.sum(k * k, axis=1) + wp2_w2(x, m) - 1.0)


def H_O_stronglyMagnetized(x, k, m):
    """O-mode in the infinitely-magnetized cold-plasma limit (default,
    valid near the polar cap):
       n^2 = (1 - wp2)/(1 - wp2 cos^2 theta_kB)
       H = 0.5*(|k|^2 - wp2*(k.Bhat)^2 - 1 + wp2)."""
    B    = Bfield(x, m)
    Bhat = B / np.linalg.norm(B, axis=1, keepdims=True)
    kdB  = np.sum(k * Bhat, axis=1)
    wp2  = wp2_w2(x, m)
    return 0.5 * (np.sum(k * k, axis=1) - wp2 * kdB**2 - 1.0 + wp2)


# =====================================================================
# Model configuration
# =====================================================================

@dataclass
class Model:
    chi: float      = np.deg2rad(30.0)
    nu_GHz: float   = 1.4              # observing/wave frequency -- FIXED for the
                                        # whole run; everything else is normalized
                                        # to it. Don't hand-edit wp2_surf/wB_surf/
                                        # omega_Rc separately to "change frequency" --
                                        # use model_from_physical() instead.
    wp2_surf: float = 0.375            # EFFECTIVE (omega_p/omega)^2 at the pole, r=1.
    wB_surf: float  = 2.0e9            # omega_B/omega at pole (B=1e12 G, 1.4 GHz).
    omega_Rc: float = 3.0e5            # omega*R_star/c at 1.4 GHz.
    r_out: float    = 300.0
    H_O: Callable   = H_O_stronglyMagnetized
    alpha_conv: float = 0.6            # alpha in eta = sqrt(2)*p*(1-p)^alpha.
    gamma0: float      = 20.0          # streaming pair-plasma Lorentz factor;
                                        # linked to wp2_surf -- use
                                        # model_from_physical() to keep consistent.
    beta0: float       = None
    freeze_on: bool = True
    r_freeze: float = 150.0

    def __post_init__(self):
        if self.beta0 is None:
            self.beta0 = np.sqrt(1.0 - 1.0 / self.gamma0**2)
        assert self.r_freeze < self.r_out, "r_freeze must lie inside r_out"

    @property
    def mhat(self):
        return np.array([np.sin(self.chi), 0.0, np.cos(self.chi)])


def model_from_physical(nu_GHz=1.4, B12=1.0, P=1.0, kappa_3=1.0, gamma0=20.0,
                         chi_deg=30.0, R_km=10.0, coeff_wp2=5598.5, **kwargs):
    """Build a Model from physical neutron-star/pulsar parameters at a
    single FIXED observing frequency, so wp2_surf, wB_surf, and omega_Rc
    stay mutually consistent (change nu_GHz/B12/P/kappa_3/gamma0 here --
    never hand-edit the derived Model fields separately).

    Scalings (all at the pole, r=1; the r-dependence away from the pole is
    handled separately by wp2_w2()/wB_w() via Bmag(x,m) ~ 1/r^3):
        omega_B/omega   = nu_B(B12)/nu_GHz,  nu_B[GHz] = 2.8e9*B12
        omega*R_star/c  = 2*pi*nu_GHz[Hz]*R_star[cm]/c
        (omega_p/omega)^2_eff = coeff_wp2*kappa_3*B12/(P*nu_GHz^2*gamma0^3)

    coeff_wp2 default (5598.5) is the first-principles Goldreich-Julian
    value: n_GJ=B/(P*c*e), omega_p^2=4*pi*(kappa*n_GJ)*e^2/m_e,
    wp2_surf=omega_p^2/(gamma0^3*omega_obs^2) -- reproduces the canonical
    n_GJ~7e10 cm^-3 at B12=P=1.

    Args:
        nu_GHz    : observing/wave frequency in GHz (one fixed value per run).
        B12       : polar surface magnetic field, in units of 1e12 G.
        P         : spin period in seconds.
        kappa_3   : pair multiplicity above Goldreich-Julian, in units of 1e3.
        gamma0    : bulk Lorentz factor of the streaming pair plasma.
        chi_deg   : magnetic obliquity in degrees.
        R_km      : stellar radius in km.
        coeff_wp2 : coefficient in the wp2_eff scaling relation; override
                    only if your own derivation uses a different convention.
        **kwargs  : forwarded to Model (e.g. r_out, alpha_conv, freeze_on,
                    r_freeze, H_O).
    Returns:
        Model instance with wp2_surf, wB_surf, omega_Rc, gamma0, nu_GHz, chi set.
    """
    nu_B_GHz = 2.8e9 * B12
    wB_surf  = nu_B_GHz / nu_GHz
    R_cm, c_cm = R_km * 1e5, 3.0e10
    omega_Rc = 2 * np.pi * (nu_GHz * 1e9) * R_cm / c_cm
    wp2_surf = coeff_wp2 * kappa_3 * B12 / (P * nu_GHz**2 * gamma0**3)
    return Model(chi=np.deg2rad(chi_deg), nu_GHz=nu_GHz, wp2_surf=wp2_surf,
                 wB_surf=wB_surf, omega_Rc=omega_Rc, gamma0=gamma0, **kwargs)


# =====================================================================
# Landau-Zener O<->X conversion
# =====================================================================

def wres_w(x, m):
    """omega_res/omega = sqrt(wp2_eff)/(1-beta0)."""
    return np.sqrt(wp2_w2(x, m)) / (1.0 - m.beta0)


def eps_param(x, m):
    """eps = omega/omega_res."""
    return 1.0 / wres_w(x, m)


def curvature_radius(x, m, rel_eps=1e-4):
    """Field-line curvature radius rho = 1/|dBhat/ds|, in R_star."""
    B    = Bfield(x, m)
    Bhat = B / np.linalg.norm(B, axis=1, keepdims=True)
    eps  = rel_eps * np.linalg.norm(x, axis=1, keepdims=True)
    Bp   = Bfield(x + eps * Bhat, m); Bp /= np.linalg.norm(Bp, axis=1, keepdims=True)
    Bm   = Bfield(x - eps * Bhat, m); Bm /= np.linalg.norm(Bm, axis=1, keepdims=True)
    kap  = np.linalg.norm(Bp - Bm, axis=1) / (2.0 * eps[:, 0])
    return 1.0 / np.maximum(kap, 1e-30)


def conversion_frame(x_ref, m, rel_eps=1e-4):
    """Local field-line curvature frame at x_ref: z0 || B, x0 along
    dBhat/ds (osculating plane, perp to z0), y0 = z0 x x0. Recomputed fresh
    at whatever position it's called with -- no persistent per-ray frame.
    Returns:
        (x0hat, y0hat, z0hat) : three (N,3) float arrays.
    """
    B   = Bfield(x_ref, m)
    z0  = B / np.linalg.norm(B, axis=1, keepdims=True)
    eps = rel_eps * np.linalg.norm(x_ref, axis=1, keepdims=True)
    Bp  = Bfield(x_ref + eps * z0, m); Bp /= np.linalg.norm(Bp, axis=1, keepdims=True)
    Bm  = Bfield(x_ref - eps * z0, m); Bm /= np.linalg.norm(Bm, axis=1, keepdims=True)
    c   = (Bp - Bm) / (2.0 * eps)
    c  -= np.sum(c * z0, axis=1, keepdims=True) * z0
    cn  = np.linalg.norm(c, axis=1, keepdims=True)
    bad = (cn[:, 0] < 1e-12)                       # straight field line (on-axis)
    if bad.any():
        alt = np.cross(z0[bad], [0.0, 0.0, 1.0])
        degen = np.linalg.norm(alt, axis=1) < 1e-8
        alt[degen] = [1.0, 0.0, 0.0]
        c[bad]  = alt
        cn[bad] = np.linalg.norm(alt, axis=1, keepdims=True)
    x0 = c / cn
    y0 = np.cross(z0, x0)
    return x0, y0, z0


def delta_LZ(x, k, x0hat, y0hat, m):
    """Delta = 2*ktx^3*rho_t*sin^3(phi) / (3*|eps^2-1|), evaluated locally
    at (x,k) using the frame (x0hat,y0hat) established at x itself."""
    ktx   = np.abs(np.sum(k * x0hat, axis=1))
    rho_t = m.omega_Rc * curvature_radius(x, m)
    B     = Bfield(x, m)
    Bx    = np.sum(B * x0hat, axis=1)
    By    = np.sum(B * y0hat, axis=1)
    sinph = np.abs(By) / np.maximum(np.sqrt(Bx**2 + By**2), 1e-30)
    eps   = eps_param(x, m)
    denom = 3.0 * np.maximum(np.abs(eps**2 - 1.0), 1e-12)
    return 2.0 * ktx**3 * rho_t * sinph**3 / denom


def eta_LZ(delta, m):
    """eta = sqrt(2)*p*(1-p)^alpha, p = exp(-Delta). Clipped to [0,1] for
    float safety only -- no extra efficiency prefactor."""
    p = np.exp(-delta)
    return np.clip(np.sqrt(2.0) * p * (1.0 - p) ** m.alpha_conv, 0.0, 1.0)


# =====================================================================
# Ray integrator
# =====================================================================

def gradH_x(x, k, m, rel_eps=1e-4):
    """dH/dx at fixed k, central differences."""
    g   = np.empty_like(x)
    eps = rel_eps * np.linalg.norm(x, axis=1)
    for i in range(3):
        dx = np.zeros_like(x); dx[:, i] = eps
        g[:, i] = (m.H_O(x + dx, k, m) - m.H_O(x - dx, k, m)) / (2.0 * eps)
    return g


def gradH_k(x, k, m, eps=1e-6):
    """dH/dk at fixed x (group-velocity direction), central differences."""
    g = np.empty_like(k)
    for i in range(3):
        dk = np.zeros_like(k); dk[:, i] = eps
        g[:, i] = (m.H_O(x, k + dk, m) - m.H_O(x, k - dk, m)) / (2.0 * eps)
    return g


def rhs(x, k, isO, m):
    """dx/ds = (dH/dk)/|dH/dk|, dk/ds = -(dH/dx)/|dH/dk|; X-modes go straight."""
    dx = np.zeros_like(k)
    dk = np.zeros_like(k)
    dx[~isO] = k[~isO]
    if isO.any():
        xo, ko = x[isO], k[isO]
        vg = gradH_k(xo, ko, m)
        vn = np.linalg.norm(vg, axis=1, keepdims=True)
        dx[isO] =  vg / vn
        dk[isO] = -gradH_x(xo, ko, m) / vn
    return dx, dk


def rk4_step(x, k, isO, h, m):
    """One RK4 step of the ray equations, per-ray step size h."""
    h1 = h[:, None]
    a1x, a1k = rhs(x,            k,             isO, m)
    a2x, a2k = rhs(x+0.5*h1*a1x, k+0.5*h1*a1k,  isO, m)
    a3x, a3k = rhs(x+0.5*h1*a2x, k+0.5*h1*a2k,  isO, m)
    a4x, a4k = rhs(x+h1*a3x,     k+h1*a3k,      isO, m)
    xn = x + h1/6.0 * (a1x + 2*a2x + 2*a3x + a4x)
    kn = k + h1/6.0 * (a1k + 2*a2k + 2*a3k + a4k)
    return xn, kn


def solve_n(x, khat, m, n_lo=1e-6, n_hi=1.0 + 1e-6, iters=48):
    """Solve H(x, n*khat) = 0 for n on the escaping (n<=1) branch.
    Returns:
        (n, ok) : (N,) float array (<=1), (N,) bool (ok=False -> evanescent).
    """
    f_lo = m.H_O(x, n_lo * khat, m)
    f_hi = m.H_O(x, n_hi * khat, m)
    ok   = (f_lo * f_hi) < 0.0
    lo   = np.full(len(x), n_lo); hi = np.full(len(x), n_hi)
    for _ in range(iters):
        mid  = 0.5 * (lo + hi)
        f_md = m.H_O(x, mid[:, None] * khat, m)
        up   = (f_md * f_lo) > 0.0
        lo   = np.where(up, mid, lo); f_lo = np.where(up, f_md, f_lo)
        hi   = np.where(up, hi, mid)
    return np.minimum(0.5 * (lo + hi), 1.0), ok


# =====================================================================
# Main propagation loop
# =====================================================================

def propagate(x, k, isO, m, t_emit=None, h_frac=0.02, max_steps=50000,
              apply_conversion=True, progress=True):
    """Integrate rays to escape (r>r_out) or absorption (r<1).
    Args:
        x, k     : (N,3) initial positions [R*] and wavevectors [omega/c].
        isO      : (N,) bool, True = O mode.
        m        : Model.
        t_emit   : (N,) initial path length/time offset (default zeros).
        h_frac   : step size as fraction of local radius. Check
                   `convergence_test` before trusting results at a given
                   h_frac -- this coherence-length rate is stiff and needs
                   smaller steps than a naive rho-based rate would.
        max_steps: safety cap; increase for smaller h_frac (steps needed
                   scale roughly as ln(r_out/r_em)/h_frac).
        apply_conversion : if True (default), mode flips are drawn and
                   applied each step, feeding back into subsequent
                   refraction (the physical run). If False, mode is held
                   fixed for the whole flight -- `nconv_exp` is still
                   accumulated, giving a DETERMINISTIC diagnostic (no RNG)
                   for convergence checks.
        progress : print a lightweight throttled status line (no external
                   dependency).
    Returns:
        dict: x, k, isO, Bfrz, t, escaped, hit, lost, nconv, nconv_exp, frozen.
        nconv_exp is the running sum of per-step probabilities (the expected
        conversion count along the as-launched trajectory), accumulated
        every run regardless of apply_conversion.
    """
    N      = len(x)
    x, k   = x.copy(), k.copy()
    isO    = isO.copy()
    active = np.ones(N, bool)
    hit    = np.zeros(N, bool)
    lost   = np.zeros(N, bool)
    frozen = np.zeros(N, bool)
    nconv     = np.zeros(N, int)
    nconv_exp = np.zeros(N, float)
    s      = np.zeros(N) if t_emit is None else np.asarray(t_emit, float).copy()
    Bfrz   = Bfield(x, m)

    def _norm(v):
        return np.linalg.norm(v, axis=1)

    def _unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _apply_flips(flip):
        """Swap mode of rays `flip` at their current position; k direction kept."""
        if flip.size == 0:
            return
        newO  = ~isO[flip]
        kh    = _unit(k[flip])
        n_new = np.ones(flip.size)
        ok    = np.ones(flip.size, bool)
        if newO.any():                                # X -> O needs a real O root
            n_O, ok_O   = solve_n(x[flip][newO], kh[newO], m)
            n_new[newO] = n_O
            ok[newO]    = ok_O
        flip, kh, n_new = flip[ok], kh[ok], n_new[ok]
        if flip.size:
            isO[flip]    = ~isO[flip]
            k[flip]      = kh * n_new[:, None]
            nconv[flip] += 1

    t_start = time.time()
    t_last  = t_start
    step = 0
    for step in range(max_steps):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        # ---------------- one RK4 step ------------------------------------
        x_old, k_old = x[idx].copy(), k[idx].copy()
        h = h_frac * _norm(x_old)
        x_new, k_new = rk4_step(x_old, k_old, isO[idx], h, m)
        x[idx], k[idx] = x_new, k_new
        s[idx] += h
        # ---------------- termination bookkeeping -------------------------
        finite = np.isfinite(x_new).all(1) & np.isfinite(k_new).all(1)
        r_new  = np.where(finite, _norm(np.nan_to_num(x_new)), np.nan)
        esc    = finite & (r_new > m.r_out)
        crash  = finite & (r_new < 1.0)
        bad    = ~finite
        done   = esc | crash | bad
        active[idx[done]] = False
        hit[idx[crash]]   = True
        lost[idx[bad]]    = True
        if progress:
            now = time.time()
            if now - t_last > 0.5:                  # throttle to ~2 updates/sec
                t_last = now
                print(f"\rstep {step+1}/{max_steps}  active={int(active.sum())}/{N}  "
                      f"conv={int(nconv.sum())}  elapsed={now-t_start:.1f}s",
                      end="", flush=True)
        keep = ~done
        al   = idx[keep]
        if al.size == 0:
            continue
        r_al = r_new[keep]
        h_al = h[keep]
        # ---------------- polarization freezing ----------------------------
        if m.freeze_on:
            frozen[al[r_al > m.r_freeze]] = True
        cm  = ~frozen[al]
        upd = al[cm]
        if upd.size == 0:
            continue
        Bfrz[upd] = Bfield(x[upd], m)            # PA reference tracks B while coupled
        # ---------------- mode conversion (Landau-Zener, local rate) -------
        # Delta/eta are smooth local quantities (no crossing/event to
        # detect). The coherence length for one passage is tilde_k_x*rho
        # (NOT rho alone -- see module docstring); eta/(tilde_k_x*rho) is
        # the local conversion rate.
        x0u, y0u, _ = conversion_frame(x[upd], m)      # frame local to THIS point
        delta = delta_LZ(x[upd], k[upd], x0u, y0u, m)
        eta   = eta_LZ(delta, m)
        rho   = curvature_radius(x[upd], m)
        ktx   = np.abs(np.sum(k[upd] * x0u, axis=1))
        l_coh = np.maximum(ktx, 1e-6) * rho
        lam   = eta / np.maximum(l_coh, 1e-30)
        P     = -np.expm1(-lam * h_al[cm])
        nconv_exp[upd] += P                     # deterministic, no RNG
        if apply_conversion:
            flip = upd[rng.random(upd.size) < P]
            _apply_flips(flip)
    if progress:
        print(f"\rstep {step+1}/{max_steps}  active={int(active.sum())}/{N}  "
              f"conv={int(nconv.sum())}  elapsed={time.time()-t_start:.1f}s")
    n_stuck = int(active.sum())
    if n_stuck:
        print(f"WARNING: {n_stuck} rays still active after {max_steps} steps "
              f"(increase max_steps or check h_frac/r_out).")
    if lost.any():
        print(f"WARNING: {int(lost.sum())} rays produced non-finite state "
              f"and were dropped.")

    escaped = ~active & ~hit & ~lost
    return dict(x=x, k=k, isO=isO, Bfrz=Bfrz, t=s, escaped=escaped, hit=hit,
                lost=lost, nconv=nconv, nconv_exp=nconv_exp, frozen=frozen)


def convergence_test(x0, k0, isO0, m, h_fracs=(0.04, 0.02, 0.01, 0.005),
                      max_steps=6000, progress=False):
    """Check step-size convergence using the DETERMINISTIC expected-conversion
    accumulator (nconv_exp, apply_conversion=False) -- no RNG draws, no mode
    feedback into the trajectory, so this isolates whether the rate integral
    itself is converging as h_frac shrinks, free of the shot noise that
    swamps the realized (stochastic) conversion count at modest N.
    Args:
        x0, k0, isO0 : launch state (same for every h_frac).
        m            : Model.
        h_fracs      : sequence of step sizes to test, largest first.
        max_steps    : passed through to propagate (increase if smaller
                       h_frac needs more steps to reach r_out -- steps
                       needed scale roughly as ln(r_out/r_em)/h_frac).
        progress     : print status lines during each propagate() call.
    Returns:
        list of dicts with keys h_frac, mean_nconv_exp.
    """
    rows = []
    for hf in h_fracs:
        r = propagate(x0.copy(), k0.copy(), isO0.copy(), m,
                      h_frac=hf, max_steps=max_steps,
                      apply_conversion=False, progress=progress)
        rows.append(dict(h_frac=hf, mean_nconv_exp=r['nconv_exp'].mean()))
        print(f"h_frac={hf:<7g}  <nconv_exp>={rows[-1]['mean_nconv_exp']:.6f}")
    return rows


# =====================================================================
# Ray launching schemes
# =====================================================================

def rotate_z_to(vecs, axis):
    """Rotate vectors so the z-axis maps onto `axis` (unit vector)."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, axis))
    if c > 1.0 - 1e-12:
        return vecs.copy()
    if c < -1.0 + 1e-12:
        return vecs * np.array([1.0, -1.0, -1.0])
    v = np.cross(z, axis)
    sn = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / sn**2)
    return vecs @ R.T


def polar_cap_dirs(N, m, deg=10.0):
    """Unit vectors over two antipodal caps of half-angle `deg` around
    m.mhat. Uses the global `rng` directly (not a default-argument
    snapshot), so re-seeding `rng` (e.g. for a cluster job) is always
    respected."""
    c   = np.cos(np.deg2rad(deg))
    n_n = N // 2
    n_s = N - n_n
    u   = np.concatenate([rng.uniform(c, 1.0, n_n),
                          rng.uniform(-1.0, -c, n_s)])
    ph  = rng.uniform(0, 2*np.pi, N)
    s   = np.sqrt(1.0 - u**2)
    d   = np.stack([s*np.cos(ph), s*np.sin(ph), u], axis=1)
    return rotate_z_to(d, m.mhat)


def launch_polar_cap(N, m, r_em=1.05, f_O=1.0, cap_deg=10.0, t_emit=None):
    """Radially outgoing rays from two antipodal polar caps, prescribed O/X mix."""
    rhat = polar_cap_dirs(N, m, deg=cap_deg)
    x    = r_em * rhat
    isO  = rng.random(N) < f_O
    n    = np.ones(N)
    good = np.ones(N, bool)
    if isO.any():
        n_O, ok_O = solve_n(x[isO], rhat[isO], m)
        n[isO]    = n_O
        good[isO] = ok_O
    k  = rhat * n[:, None]
    te = np.zeros(N) if t_emit is None else np.asarray(t_emit, float)
    dropped = N - int(good.sum())
    if dropped:
        print(f"launch_polar_cap: dropped {dropped}/{N} evanescent O-mode rays "
              f"({100*dropped/N:.1f}%).")
    return x[good], k[good], isO[good], te[good]


def launch_radial(N, m, r_em=1.05, f_O=1.0, t_emit=None):
    """Isotropic shell of radially outgoing rays, prescribed O/X mix."""
    u    = rng.uniform(-1, 1, N)
    ph   = rng.uniform(0, 2*np.pi, N)
    s    = np.sqrt(1 - u**2)
    rhat = np.stack([s*np.cos(ph), s*np.sin(ph), u], axis=1)
    x    = r_em * rhat
    isO  = rng.random(N) < f_O
    n    = np.ones(N)
    good = np.ones(N, dtype=bool)
    if isO.any():
        n_O, ok_O = solve_n(x[isO], rhat[isO], m)
        n[isO]    = n_O
        good[isO] = ok_O
    k  = rhat * n[:, None]
    te = np.zeros(N) if t_emit is None else np.asarray(t_emit, float)
    return x[good], k[good], isO[good], te[good]


def launch_field_aligned(N, m, r_em=1.05, f_O=0.5, cap_deg=10.0, t_emit=None):
    """Rays launched exactly TANGENT to the local dipole field line at each
    emission point (a discharge streaming along B) -- NOT radially
    outward, and with no additional angular spread: k_hat = Bhat(x_em)
    exactly.

    tilde_k_x is therefore exactly 0 at emission for every ray; it becomes
    nonzero purely through subsequent O-mode refraction along the ray.

    cap_deg is just where on the star the discharge footpoints are drawn
    from (colatitude spread around the magnetic axis) -- it can be any
    size; waves propagate along any field line and there is no
    open/closed-field-line restriction imposed here.

    Args:
        N       : int -- rays to draw (returned M<=N after evanescence cut).
        m       : Model -- uses m.mhat, m.H_O.
        r_em    : float -- emission radius in R_star.
        f_O     : float -- fraction launched as O-mode (rest X).
        cap_deg : float -- footpoint half-angle (deg) around the magnetic axis.
        t_emit  : (N,) float array or None -- emission times (default zeros).
    Returns:
        (x, k, isO, t_emit) : (M,3), (M,3), (M,), (M,).
    """
    rhat_foot = polar_cap_dirs(N, m, deg=cap_deg)
    x = r_em * rhat_foot

    # Local field direction at each footpoint is the launch direction
    # exactly. Bfield() alternates sign by hemisphere (dipole lines enter
    # at one pole, leave at the other) -- flip it to always point outward
    # (away from the star), the physical streaming direction, regardless
    # of which magnetic pole the footpoint is near.
    khat = Bfield(x, m)
    khat = khat / np.linalg.norm(khat, axis=1, keepdims=True)
    inward = np.sum(khat * rhat_foot, axis=1) < 0.0
    khat[inward] *= -1.0

    isO  = rng.random(N) < f_O
    n    = np.ones(N)
    good = np.ones(N, bool)
    if isO.any():
        n_O, ok_O = solve_n(x[isO], khat[isO], m)
        n[isO]    = n_O
        good[isO] = ok_O
    k  = khat * n[:, None]
    te = np.zeros(N) if t_emit is None else np.asarray(t_emit, float)
    dropped = N - int(good.sum())
    if dropped:
        print(f"launch_field_aligned: dropped {dropped}/{N} evanescent O-mode rays "
              f"({100*dropped/N:.1f}%).")
    return x[good], k[good], isO[good], te[good]


def launch_polar_cap_binned(n_per_bin, m, colat_edges_deg, r_em=1.05, f_O=0.5):
    """Launch mixed O/X rays in colatitude annuli around the magnetic axis
    (radial launch direction), for geometry-dependence studies.
    Args:
        n_per_bin      : int -- rays drawn per colatitude bin (before evanescence cut).
        m              : Model.
        colat_edges_deg: 1D array of bin edges in degrees from the magnetic
                         axis (e.g. [0,5,10,...]); north+south caps both populated.
        r_em           : float -- emission radius in R_star.
        f_O            : float -- fraction launched as O-mode (rest X).
    Returns:
        (x, k, isO, bin_idx, edges) : (M,3),(M,3),(M,) bool,(M,) int, edges array.
    """
    edges = np.asarray(colat_edges_deg, float)
    xs, ks, isOs, bins_ = [], [], [], []
    for i in range(len(edges) - 1):
        c_lo, c_hi = edges[i], edges[i + 1]
        N = n_per_bin
        u_lo, u_hi = np.cos(np.deg2rad(c_hi)), np.cos(np.deg2rad(c_lo))
        n_n = N // 2; n_s = N - n_n
        u  = np.concatenate([rng.uniform(u_lo, u_hi, n_n),
                             rng.uniform(-u_hi, -u_lo, n_s)])
        ph = rng.uniform(0, 2*np.pi, N)
        s  = np.sqrt(np.clip(1.0 - u**2, 0.0, None))
        d  = np.stack([s*np.cos(ph), s*np.sin(ph), u], axis=1)
        rhat = rotate_z_to(d, m.mhat)
        x    = r_em * rhat
        isO  = rng.random(N) < f_O
        n    = np.ones(N); good = np.ones(N, bool)
        if isO.any():
            n_O, ok_O = solve_n(x[isO], rhat[isO], m)
            n[isO]    = n_O
            good[isO] = ok_O
        k = rhat * n[:, None]
        xs.append(x[good]); ks.append(k[good]); isOs.append(isO[good])
        bins_.append(np.full(int(good.sum()), i))
    return (np.concatenate(xs), np.concatenate(ks), np.concatenate(isOs),
            np.concatenate(bins_), edges)


def geometry_sweep(m, n_per_bin=3000, colat_edges_deg=None, f_O=0.5,
                    h_frac=0.01, max_steps=4000, progress=False):
    """Run the mixed O/X ensemble across colatitude bins and report the
    outgoing (escaped) mode fractions per bin.
    Args:
        m               : Model.
        n_per_bin       : rays drawn per bin (before evanescence cut).
        colat_edges_deg : bin edges in degrees; default 0-35 in 5-deg steps.
        f_O             : launched O-mode fraction.
        h_frac, max_steps, progress : passed to propagate().
    Returns:
        (rows, res, bin_idx) : rows is a list of per-bin dicts; res/bin_idx
        are the raw propagate() output and per-ray bin index.
    """
    if colat_edges_deg is None:
        colat_edges_deg = np.linspace(0, 35, 8)
    x, k, isO, bin_idx, edges = launch_polar_cap_binned(
        n_per_bin, m, colat_edges_deg, f_O=f_O)
    res = propagate(x, k, isO, m, h_frac=h_frac, max_steps=max_steps,
                     progress=progress)
    rows = []
    for i in range(len(edges) - 1):
        sel = (bin_idx == i) & res['escaped']
        n_tot = int(sel.sum())
        if n_tot == 0:
            rows.append(dict(colat_lo=edges[i], colat_hi=edges[i+1], n_escaped=0,
                              f_O_out=np.nan, f_X_out=np.nan, mean_nconv=np.nan))
            continue
        f_O_out = float(res['isO'][sel].mean())
        rows.append(dict(colat_lo=edges[i], colat_hi=edges[i+1], n_escaped=n_tot,
                          f_O_out=f_O_out, f_X_out=1.0 - f_O_out,
                          mean_nconv=float(res['nconv'][bin_idx == i].mean())))
    return rows, res, bin_idx


# =====================================================================
# I/O
# =====================================================================

def save_dict_to_hdf5(dictionary, filename):
    """Save a flat dict of arrays (e.g. a propagate() output) to a
    gzip-compressed HDF5 file."""
    with h5py.File(filename, "w", libver="latest") as h5f:
        for name, data in dictionary.items():
            arr = np.asarray(data)
            chunks = ((min(arr.shape[0], max(1, int(1e6 // arr.itemsize))),)
                      if arr.ndim == 1 else None)
            h5f.create_dataset(name, data=arr, compression="gzip",
                                compression_opts=4, chunks=chunks, shuffle=True)
    print(f"Saved {len(dictionary)} datasets to {filename}")


# =====================================================================
# Command-line entry point
# =====================================================================

def build_argparser():
    p = argparse.ArgumentParser(
        description="Pulsar magnetosphere O<->X mode-conversion ray tracer")
    # Neutron-star / frequency parameters -> model_from_physical()
    p.add_argument("--nu-GHz", type=float, default=1.4, help="observing frequency [GHz]")
    p.add_argument("--B12", type=float, default=1.0, help="polar B field [1e12 G]")
    p.add_argument("--P", type=float, default=1.0, help="spin period [s]")
    p.add_argument("--kappa3", type=float, default=1.0, help="pair multiplicity [1e3]")
    p.add_argument("--gamma0", type=float, default=20.0, help="streaming Lorentz factor")
    p.add_argument("--chi-deg", type=float, default=30.0, help="magnetic obliquity [deg]")
    # Launch parameters -> launch_field_aligned()
    p.add_argument("--N", type=int, default=1_000_000, help="number of rays to launch")
    p.add_argument("--r-em", type=float, default=1.5, help="emission radius [R_star]")
    p.add_argument("--f-O", type=float, default=0.9, help="launched O-mode fraction")
    p.add_argument("--cap-deg", type=float, default=20.0, help="footpoint half-angle [deg]")
    # Propagation parameters -> propagate()
    p.add_argument("--h-frac", type=float, default=0.002, help="RK4 step size (fraction of r)")
    p.add_argument("--max-steps", type=int, default=50_000)
    # Misc
    p.add_argument("--seed", type=int, default=42, help="RNG seed (vary per cluster job)")
    p.add_argument("--out", type=str, default="res_out.h5", help="output HDF5 path")
    p.add_argument("--no-progress", action="store_true", help="disable status printing")
    return p


def main():
    args = build_argparser().parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    m = model_from_physical(nu_GHz=args.nu_GHz, B12=args.B12, P=args.P,
                             kappa_3=args.kappa3, gamma0=args.gamma0,
                             chi_deg=args.chi_deg)

    x0, k0, isO0, te = launch_field_aligned(
        args.N, m, r_em=args.r_em, f_O=args.f_O, cap_deg=args.cap_deg)

    res = propagate(x0, k0, isO0, m, t_emit=te, h_frac=args.h_frac,
                     max_steps=args.max_steps, progress=not args.no_progress)

    print(f"escaped: {res['escaped'].sum()}, hit star: {res['hit'].sum()}, "
          f"mean conversions/ray: {res['nconv'].mean():.3f}, "
          f"final O fraction: {res['isO'][res['escaped']].mean():.3f}")

    save_dict_to_hdf5(res, args.out)


if __name__ == "__main__":
    main()
