"""
    pepo_to_2d(O; virtual_order=:WNES, is_fermion=nothing)

Close the physical legs of a uniform PEPO and return a tensor for `LoopTNR`.
The default virtual order `(west, north, east, south)` is the convention used
by ThermalPEPOKit's Ising, XXZ and spinless-fermion PEPOs. Use `:NESW` for
TNRKit's native thermal tensors, including `classical_ising_3D`.

Fermionic braiding is detected automatically. The extra physical twist cancels
the supertrace so that the physical legs take the ordinary thermal trace.
Bond arrows are then brought to LoopTNR's outgoing/outgoing/incoming/incoming
convention with inverse flips at the two ends of each bond.
"""
function pepo_to_2d(
        O::TNOTensor; virtual_order::Symbol = :WNES,
        is_fermion::Union{Nothing, Bool} = nothing
    )
    _check_thermal_order(virtual_order)
    A = virtual_order == :NESW ? permute(O, ((1, 2), (6, 3, 4, 5))) : O
    fermionic = isnothing(is_fermion) ?
        BraidingStyle(sectortype(A)) == Fermionic() : is_fermion
    A = fermionic ? twist(A, 2) : A
    m = id(eltype(A), space(A, 1))
    @tensor T[-1 -2; -4 -3] := A[1 2; -1 -2 -3 -4] * m[2; 1]

    outgoing = Tuple(i for i in 1:2 if isdual(space(T, i)))
    incoming = Tuple(i for i in 3:4 if !isdual(space(T, i)))
    isempty(outgoing) || (T = flip(T, outgoing; inv = true))
    isempty(incoming) || (T = flip(T, incoming; inv = false))
    return T
end

function _check_thermal_order(order)
    return order in (:WNES, :NESW) ? nothing :
        throw(ArgumentError("virtual_order must be :WNES or :NESW"))
end

function _thermal_reorder(tno::TNO, indices)
    return TNO(map(A -> permute(A, ((1, 2), indices)), tno.A))
end

function _thermal_step!(scheme, fixed_layer, trunc, virtual_order)
    if virtual_order == :NESW
        return step!(scheme, fixed_layer, trunc)
    end
    canonical = ThermalTNR(_thermal_reorder(scheme.T, (4, 5, 6, 3)))
    layer = _thermal_reorder(fixed_layer, (4, 5, 6, 3))
    step!(canonical, layer, trunc)
    scheme.T = _thermal_reorder(canonical.T, (6, 3, 4, 5))
    return scheme
end

# A local graded trace can cancel in a fermionic network. Frobenius scaling
# avoids that cancellation; retaining every factor preserves the partition sum.
function _normalize_thermal!(scheme::ThermalTNR)
    n = norm(scheme.T[1, 1])
    isfinite(n) && n > 0 || throw(ArgumentError("PEPO norm must be positive and finite"))
    scheme.T[1, 1] /= n
    return n
end

function _thermal_identity(O::TNOTensor)
    P = id(eltype(O), space(O, 1))
    units = map(i -> ones(eltype(O), oneunit(space(O, i))), 3:6)
    @tensor A[-1 -2; -3 -4 -5 -6] := P[-1; -2] *
        units[1][-3] * units[2][-4] * units[3][-5] * units[4][-6]
    return A
end

function _thermal_schedule(beta_list, beta_pivot; beta_atol, beta_rtol)
    all(x -> isfinite(x) && x >= 0, (beta_atol, beta_rtol)) ||
        throw(ArgumentError("beta_atol and beta_rtol must be finite and nonnegative"))
    isempty(beta_list) && throw(ArgumentError("beta_list must not be empty"))
    all(x -> isfinite(x) && x >= 0, beta_list) ||
        throw(ArgumentError("beta_list must contain finite, nonnegative beta values"))
    all(>(0), diff(beta_list)) ||
        throw(ArgumentError("beta_list must be strictly increasing"))

    # The supplied PEPO is one fixed step, not an already evolved initial state.
    # An optional zero explicitly labels the identity before the first step.
    offset = iszero(first(beta_list)) ? 1 : 0
    if length(beta_list) > offset
        delta = beta_list[offset + 1]
        for (i, beta) in enumerate(beta_list)
            isapprox(beta, (i - offset) * delta; atol = beta_atol, rtol = beta_rtol) ||
                throw(ArgumentError(
                    "beta_list must be a uniform grid delta_beta, 2delta_beta, ... " *
                        "with an optional leading zero; the PEPO represents one delta_beta step"
                ))
        end
    end

    indices = Int[]
    for beta in beta_pivot
        isfinite(beta) && beta >= 0 ||
            throw(ArgumentError("beta_pivot must contain finite, nonnegative beta values"))
        matches = findall(b -> isapprox(b, beta; atol = beta_atol, rtol = beta_rtol), beta_list)
        isempty(matches) && throw(ArgumentError("beta_pivot value $beta has no match in beta_list"))
        length(matches) == 1 || throw(ArgumentError(
            "beta_pivot value $beta matches several grid points; reduce beta_atol/beta_rtol"
        ))
        index = only(matches)
        index in indices && throw(ArgumentError("beta_pivot selects the same grid point more than once"))
        push!(indices, index)
    end
    return offset, indices
end

function _thermal_loop_cft(
        O; virtual_order, is_fermion, χ_tnr, maxrg, loop_parameters, shape,
        fast_tau_alg, quiet, verbosity
    )
    T = pepo_to_2d(O; virtual_order, is_fermion)
    norm(imag(T)) < 1.0e-14 && (T = real(T))
    xy = LoopTNR(T)
    valid = Bool[]
    function measure!(state)
        n = finalize!(state)
        converged = Ref(true)
        logger = LoggingExtras.ActiveFilteredLogger(Base.CoreLogging.current_logger()) do log
            failed = occursin("Solver did not converge", string(log.message)) ||
                occursin("CFTTransferMatrix eigensolver did not converge", string(log.message))
            failed && (converged[] = false)
            return !(quiet && failed)
        end
        cft = Base.CoreLogging.with_logger(logger) do
            CFTData(state; shape, fast_tau_alg)
        end
        push!(valid, converged[] && isfinite(cft.central_charge) &&
            isfinite(cft.modular_parameter) && all(isfinite, cft.scaling_dimensions))
        return n, cft
    end

    # run!(..., maxiter(0)) in the general API still performs one iteration.
    # Handle step zero explicitly so maxrg=0 really means no coarse graining.
    first_measurement = measure!(xy)
    history = [first_measurement]
    if maxrg > 0
        append!(history, run!(
            xy, truncrank(χ_tnr), maxiter(maxrg), loop_parameters,
            Finalizer(measure!, Tuple{typeof(first_measurement[1]), CFTData});
            finalize_beginning = false, verbosity
        ))
    end
    return first.(history), last.(history), valid
end

"""
    thermal_cft(pepo, beta_list, beta_pivot=beta_list; kwargs...)
    run!(scheme::ThermalTNR, beta_list, beta_pivot; kwargs...)

Evolve a uniform PEPO from the identity at **beta = 0**, adding the same local
imaginary-time layer at each point of `beta_list`. Run LoopTNR only at the
points selected by `beta_pivot`, matched with `isapprox`.

`pepo` represents **one** `delta_beta` step. Supply a uniform grid
`delta_beta:delta_beta:beta_max`, optionally including an initial zero. The
step size must agree with the one used to construct the PEPO; it cannot be
inferred from tensor entries. Sparse observation points belong in `beta_pivot`,
not in `beta_list`. A pivot at zero measures the physical identity.

`thermal_cft` copies its input. The `run!` form treats `scheme.T` as the fixed
step layer on entry and leaves it at the last point in `beta_list`. Only a
1x1 unit cell is supported by this CFT driver. The lower-level ThermalTNR
stacking API continues to support larger unit cells.

Return one named tuple per pivot, in **beta_pivot order**, with fields:
- `beta`, `depth`: matched beta and number of accumulated layers (zero at beta=0).
- `cft_history`: `CFTData` at RG steps `0:maxrg`; access step `k` at index `k+1`.
- `cft_valid`: finite data and converged CFT eigensolver/modular-parameter fit.
- `norms_τ`, `norms_xy`: all removed time and in-plane normalization factors.
- `log_partition_per_site`: approximate log Z per in-plane site, including both
  normalization histories. For a quantum thermal PEPO, `f = -logZ / beta`.

# Keywords
- `χ_τ=12`, `trunc=truncrank(χ_τ)`: imaginary-time bond truncation.
- `χ_tnr=16`, `maxrg=8`, `max_loop=12`: LoopTNR bond dimension, RG steps and sweeps.
- `loop_parameters=LoopParameters(sweeping=maxiter(max_loop), nuclear_norm=true)`:
  override the loop optimizer settings.
- `virtual_order=:WNES`: ThermalPEPOKit convention; use `:NESW` for native TNRKit tensors.
- `is_fermion=nothing`: automatic physical trace convention; a Bool overrides it.
- `shape=[sqrt(2), 2sqrt(2), 0]`, `fast_tau_alg=true`: CFT extraction geometry/fit.
- `beta_atol=1e-12`, `beta_rtol=1e-10`: grid and pivot matching tolerances.
- `progress=false`: set `true` to show a ProgressMeter bar over all points in `beta_list`.
- `quiet=false`, `verbosity=0`: optionally silence failed-fit warnings, which
  are still reflected in `cft_valid`, and control TNRKit progress logging.

The PEPO is normalized after every time step using its Frobenius norm to avoid
fermionic graded-trace cancellations. Measuring a pivot does not change the
evolution trajectory. CFT fits away from a critical point can fail; consult
`cft_valid` and compare intermediate RG steps and increasing bond dimensions.
"""
function thermal_cft(pepo::TNOTensor, beta_list::AbstractVector{<:Real}, beta_pivot = beta_list; kwargs...)
    return run!(ThermalTNR(copy(pepo)), beta_list, beta_pivot; kwargs...)
end

function thermal_cft(scheme::ThermalTNR, beta_list::AbstractVector{<:Real}, beta_pivot = beta_list; kwargs...)
    return run!(copy(scheme), beta_list, beta_pivot; kwargs...)
end

function run!(
        scheme::ThermalTNR, beta_list::AbstractVector{<:Real}, beta_pivot;
        χ_τ::Int = 12, trunc::TruncationStrategy = truncrank(χ_τ),
        χ_tnr::Int = 16, maxrg::Int = 8, max_loop::Int = 12,
        loop_parameters::LoopParameters = LoopParameters(;
            sweeping = maxiter(max_loop), nuclear_norm = true
        ),
        virtual_order::Symbol = :WNES, is_fermion::Union{Nothing, Bool} = nothing,
        shape = [sqrt(2), 2 * sqrt(2), 0], fast_tau_alg::Bool = true,
        beta_atol::Real = 1.0e-12, beta_rtol::Real = 1.0e-10,
        progress::Bool = false, quiet::Bool = false, verbosity::Int = 0
    )
    size(scheme.T) == (1, 1) || throw(ArgumentError("thermal_cft supports a 1x1 PEPO unit cell"))
    _check_thermal_order(virtual_order)
    χ_τ > 0 && χ_tnr > 0 || throw(ArgumentError("bond dimensions must be positive"))
    maxrg >= 0 || throw(ArgumentError("maxrg must be nonnegative"))
    max_loop > 0 || throw(ArgumentError("max_loop must be positive"))
    offset, indices = _thermal_schedule(beta_list, beta_pivot; beta_atol, beta_rtol)
    fixed_layer = copy(scheme.T)
    canonical_layer = virtual_order == :NESW ? fixed_layer :
        _thermal_reorder(fixed_layer, (4, 5, 6, 3))
    _check_tno_compatibility(canonical_layer, canonical_layer)

    result = Vector{NamedTuple}(undef, length(indices))
    observation = Dict(index => position for (position, index) in enumerate(indices))
    norms_τ = Float64[]
    ProgressMeter.@showprogress enabled = progress desc = "β evolution: " for (i, beta) in enumerate(beta_list)
        depth = i - offset
        if depth == 0
            scheme.T = TNO(_thermal_identity(fixed_layer[1, 1]))
        elseif depth == 1
            scheme.T = copy(fixed_layer)
            push!(norms_τ, _normalize_thermal!(scheme))
        else
            _thermal_step!(scheme, fixed_layer, trunc, virtual_order)
            push!(norms_τ, _normalize_thermal!(scheme))
        end
        haskey(observation, i) || continue
        norms_xy, cft_history, cft_valid = _thermal_loop_cft(
            scheme.T[1, 1]; virtual_order, is_fermion, χ_tnr, maxrg,
            loop_parameters, shape, fast_tau_alg, quiet, verbosity
        )
        log_partition_per_site = sum(log, norms_τ; init = 0.0) - free_energy(norms_xy, 1.0)
        result[observation[i]] = (;
            beta, depth, cft_history, cft_valid, norms_τ = copy(norms_τ),
            norms_xy, log_partition_per_site
        )
    end
    return result
end
