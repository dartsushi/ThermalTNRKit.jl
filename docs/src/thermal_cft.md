# Thermal PEPO to LoopTNR CFT data

`thermal_cft` combines imaginary-time PEPO evolution and LoopTNR measurements.
It uses the physical trace and leg convention of the Ising, XXZ and spinless
fermion examples in ThermalPEPOKit. It detects fermionic braiding automatically.

The local fork still has the Julia package name **TNRKit**. To select this
checkout from your Julia project, run this once before loading TNRKit:

```julia
using Pkg
Pkg.develop(path="ThermalTNRKit.jl")  # from the ThermalPEPOKit directory
```

Start a fresh Julia session if another TNRKit version is already loaded.

## Evolution and observation grids

The calculation starts with the identity at **β = 0**. The supplied tensor `O`
is one fixed imaginary-time layer, approximating `exp(-delta_beta * H)`.
`beta_list` is the complete, uniform evolution grid; `beta_pivot` contains
only the points where LoopTNR should run.

```julia
using TNRKit, TensorKit

delta_beta = 0.05                    # must match the construction of O
beta_list = 0.0:delta_beta:4.0        # the leading zero is optional
beta_pivot = 2.0:0.2:4.0

rows = thermal_cft(O, beta_list, beta_pivot;
    χ_τ=12, χ_tnr=16, maxrg=8, max_loop=12)

# CFTData contains central_charge, modular_parameter and scaling_dimensions.
rg_step = 4
betas = [row.beta for row in rows]
central_charges = [real(row.cft_history[rg_step + 1].central_charge) for row in rows]
valid = [row.cft_valid[rg_step + 1] for row in rows]
spectra = [row.cft_history[rg_step + 1].scaling_dimensions for row in rows]
```

There are exactly `maxrg + 1` CFT entries per pivot: the traced input tensor at
RG step 0, then one entry after each LoopTNR step. Pivot matching tolerates
floating-point rounding. Missing or ambiguous pivots, duplicate observations
and a nonuniform evolution grid are rejected before evolution begins. Results
follow `beta_pivot` order, even if those points were supplied out of order.

Pass `progress=true` to display a ProgressMeter bar over the beta loop,
including evolution points without CFT measurements. The default is
`progress=false`. `quiet` and `verbosity` control solver warnings and TNRKit
logging separately.

For an Ising or XXZ PEPO constructed by ThermalPEPOKit, pass its tensor as `O`.
For a spinless-fermion PEPO, use the same call without extra trace options:

```julia
O = saved["O"]
rows = thermal_cft(O, 0.0:0.01:4.0, 2.0:0.2:4.0;
    χ_τ=12, χ_tnr=16, maxrg=8, max_loop=12)
```

The package cannot recover `delta_beta` from tensor entries. Use the beta
stored with the PEPO when choosing the grid. The first positive grid point
always represents **one** layer; the tensor is never labeled as beta zero.

## Leg conventions and normalization

The default `virtual_order=:WNES` means `(west, north, east, south)`, as used by
ThermalPEPOKit. The driver permutes into the TTNR kernel's `(north, east,
south, west)` convention before stacking and restores the supplied convention
afterward. All permutations and flips use TensorKit to preserve fermionic signs.
`pepo_to_2d` closes the physical legs with an ordinary trace and adjusts the
virtual arrows for LoopTNR.

For native TNRKit tensors, select `:NESW` explicitly:

```julia
O = classical_ising_3D(0.2)
rows = thermal_cft(O, 0:4, [3, 4]; virtual_order=:NESW,
    χ_τ=4, χ_tnr=4, maxrg=2, max_loop=2)
```

Here the grid labels slab depth; `0` is the identity and positive integers
count layers. The physical inverse temperature of the classical model is
separate from that depth.

Each time step removes the local Frobenius norm. This avoids dividing by a
nearly vanishing local fermionic graded trace. Every scale factor is retained
in `row.norms_τ`; LoopTNR scales are retained in `row.norms_xy`. Measuring a
pivot uses a separate LoopTNR state and does not alter subsequent evolution.

`row.log_partition_per_site` combines both histories into an approximate
logarithm of the partition function per in-plane site. For a quantum thermal
PEPO at positive beta:

```julia
free_energies = [-row.log_partition_per_site / row.beta for row in rows]
```

`thermal_cft` leaves its input unchanged. To retain the evolved PEPO, use the
mutating form; its final tensor has the same virtual order as the input:

```julia
scheme = ThermalTNR(O)
rows = run!(scheme, beta_list, beta_pivot; χ_τ=12, χ_tnr=16)
evolved_pepo = scheme.T[1, 1]
```

This driver currently supports a uniform 1×1 PEPO unit cell.

## Fit validity

`cft_valid[k]` is false when the CFT eigensolver or modular-parameter solver
reports nonconvergence, or a returned value is nonfinite. Raw `CFTData` is
retained for inspection. `quiet=true` suppresses those fit warnings without
changing the validity flags. Other warnings and exceptions remain visible.

A valid fit alone does not establish criticality or convergence in bond
dimension. Inspect the flow across RG steps and repeat with larger `χ_τ` and
`χ_tnr` before interpreting a central charge or scaling dimension.

```@docs
thermal_cft
pepo_to_2d
```
