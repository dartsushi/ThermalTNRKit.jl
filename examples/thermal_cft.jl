# Run with: julia --project=ThermalTNRKit.jl ThermalTNRKit.jl/examples/thermal_cft.jl
using TNRKit, TensorKit

# A small native-tensor example that needs no saved data. Here beta_list labels
# slab depth; the classical inverse temperature is the argument to the model.
O = classical_ising_3D(0.2)
rows = thermal_cft(O, 0:4, [2, 4]; virtual_order = :NESW,
    χ_τ = 4, χ_tnr = 4, maxrg = 2, max_loop = 2)

for row in rows
    println("depth = $(row.depth), c by RG step = ",
        [real(cft.central_charge) for cft in row.cft_history],
        "; valid fits = ", row.cft_valid)
end

# For a ThermalPEPOKit Ising, XXZ or spinless-fermion tensor, the same call is:
#
# delta_beta = 0.05                  # the step used to construct your PEPO
# O = ...                           # tensor for exp(-delta_beta * H)
# beta_list = 0.0:delta_beta:4.0      # full evolution grid, starting at zero
# beta_pivot = 2.0:0.2:4.0           # selected LoopTNR measurements
# rows = thermal_cft(O, beta_list, beta_pivot;
#     χ_τ=12, χ_tnr=16, maxrg=8, max_loop=12)
# c = [real(row.cft_history[5].central_charge) for row in rows]  # RG step 4
# valid = [row.cft_valid[5] for row in rows]
#
# W,N,E,S virtual order and the fermionic physical trace are the defaults.
