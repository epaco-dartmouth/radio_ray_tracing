# Pulsar Magnetospheric Ray-Tracing & Polarization Simulator

A Hamiltonian ray-tracing code for modeling radio wave propagation through a
pulsar's magnetized plasma magnetosphere, including O-mode/X-mode refraction,
mode conversion (linear or Landau-Zener), polarization-angle evolution, and
polarization freezing — producing synthetic pulse profiles and polarization
maps directly comparable to radio pulsar observations.

## Physics Overview

Radio emission generated near a pulsar's polar cap propagates outward through
a highly magnetized, relativistically streaming plasma. Two wave modes exist
in this medium:

- **O-mode (ordinary)**: refracts due to the plasma, and can convert into the
  X-mode near a resonance layer.
- **X-mode (extraordinary)**: propagates as if in vacuum (`|k| = 1`), i.e.
  effectively unaffected by the plasma to lowest order in this model.

The code integrates the **Hamiltonian ray equations**

```
dx/ds = ∂H/∂k / |∂H/∂k|,     dk/ds = -∂H/∂x / |∂H/∂k|
```

for a cold, strongly magnetized plasma dispersion relation, with the plasma
density and magnetic field set by a **dipole model** normalized to the
stellar surface.

### Key physical ingredients

| Component | Description |
|---|---|
| `Bfield`, `Bmag` | Dipole magnetic field geometry (`m.mhat` sets the magnetic axis) |
| `wp2_w2` | Effective (relativistically suppressed) plasma frequency squared, ∝ \|B\| |
| `H_O_isotropic` / `H_O_stronglyMagnetized` | O-mode dispersion relations (unmagnetized vs. strongly magnetized limits) |
| `wres_w`, `eps_param` | Cyclotron resonance layer location, used for mode conversion |
| `curvature_radius`, `conversion_frame` | Local field-line geometry used in the Landau-Zener (LZ) conversion model |
| `delta_LZ`, `eta_LZ` | Landau-Zener O↔X conversion probability |
| `adiabaticity`, `conversion_probability` | Simplified "linear" conversion model (alternative to LZ) |
| `propagate` | Main integrator: refraction + stochastic mode conversion + polarization freezing |
| `make_maps`, `plot_maps`, `pulse_profile` | Post-processing into Stokes I/Q/U sky maps and single-observer pulse profiles |

### Mode conversion models (`Model.conv_model`)

- `"LZ"` — Landau-Zener theory, tied to local curvature radius and the
  wave/resonance frequency ratio (`delta_LZ`, `eta_LZ`).
- `"linear"` — a simplified adiabaticity-based swap probability
  (`adiabaticity`, `conversion_probability`); faster, less physically
  detailed.
- `"none"` — disables conversion; rays keep their launch mode.

### Polarization freezing

Beyond a radius `m.r_freeze`, the polarization angle (PA) reference field
`Bfrz` is frozen at its last value and conversion is disabled
(`m.freeze_on = True/False` toggles this behavior). This reflects the
physical expectation that polarization stops adiabatically tracking the
local field once the wave decouples from the plasma.

## Repository Structure

```
.
├── README.md
├── ray_tracing.ipynb        # Main notebook (model, integrator, analysis)
├── environment.yml          # (optional) conda/pip environment spec
└── figures/                 # Example output plots (optional)
```

## Installation

```bash
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>
pip install numpy matplotlib tqdm
```

or, with conda:

```bash
conda env create -f environment.yml
conda activate raytracing
```

## Usage

Open the notebook:

```bash
jupyter lab ray_tracing.ipynb
```

### Minimal example

```python
M = Model(chi=np.deg2rad(30), wp2_surf=0.375, conv_model="linear")

x0, k0, isO0, te = launch_polar_cap(N=200_000, m=M, cap_deg=35.0, f_O=1.0)
res = propagate(x0, k0, isO0, M, t_emit=te)

maps = make_maps(res)
L, PA, Xf = plot_maps(maps)

pulse_profile(maps, zeta_deg=25)
```

### Key `Model` parameters

| Parameter | Meaning |
|---|---|
| `chi` | Magnetic obliquity (angle between rotation and magnetic axes) |
| `wp2_surf` | Effective (γ⁻³-suppressed) plasma frequency² at the magnetic pole |
| `wB_surf` | Cyclotron frequency ratio ω_B/ω at the pole |
| `gamma0` | Bulk Lorentz factor of the streaming plasma (must be consistent with `wp2_surf`) |
| `r_out` | Outer integration boundary (in R★), where rays are considered "escaped" |
| `conv_model` | `"LZ"`, `"linear"`, or `"none"` |
| `r_freeze` | Radius beyond which PA and conversion freeze |

## Output

- **Stokes maps** (`make_maps`): Intensity, linear polarization fraction, PA,
  and X-mode fraction, binned over rotation phase and line-of-sight
  colatitude (`cos θ`).
- **Pulse profiles** (`pulse_profile`): 1D slices through the sky maps at a
  fixed observer viewing angle ζ, mimicking what a distant observer sees as
  the star rotates.

## Physical Caveats / Assumptions

- Cold plasma dispersion relation; wave damping and absorption are not
  modeled beyond a hard stellar-surface cutoff (`r < 1` ⇒ ray "hits" the
  star).
- The dipole field is static and axisymmetric; no rotational sweepback or
  retardation effects are included.
- `wp2_surf` and `gamma0` are linked through the assumed relativistic
  suppression scaling; changing one without the other breaks physical
  consistency (see inline comment in `Model`).

