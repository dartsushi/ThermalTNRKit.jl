using Test, TNRKit, TensorKit, LinearAlgebra

# An exactly soluble on-site Hamiltonian tests beta labels, ordinary thermal
# traces and the reconstruction of log Z independently of the RG algorithm.
function onsite_pepo(P, delta; energies = [0.3, 1.1], scalar = Float64)
    M = TensorMap(scalar.(diagm(exp.(-delta .* energies))), P ← P)
    V = oneunit(P)
    w, n, e, s = ones(scalar, V'), ones(scalar, V), ones(scalar, V), ones(scalar, V')
    @tensor O[-1 -2; -3 -4 -5 -6] := M[-1; -2] * w[-3] * n[-4] * e[-5] * s[-6]
    return O
end

@testset "Thermal beta grid and pivot validation" begin
    O = onsite_pepo(ℂ^2, 0.1)
    for (grid, pivots) in (
            (Float64[], [0.1]), ([0.1, 0.3], [0.1]),
            ([0.1, 0.1], [0.1]), ([-0.1, 0.0], [0.0]),
            ([0.0, Inf], [0.0]), ([0.1, 0.2], [0.3]),
            ([0.1, 0.2], [NaN]), ([0.1, 0.2], [0.1, 0.1])
        )
        scheme = ThermalTNR(copy(O))
        @test_throws ArgumentError run!(scheme, grid, pivots; maxrg = 0)
        @test scheme.T[1, 1] == O # Invalid requests are rejected before mutation.
    end
    @test_throws ArgumentError thermal_cft(O, [0.1], [0.1]; maxrg = -1)
    @test_throws ArgumentError thermal_cft(O, [0.1], [0.1]; max_loop = 0)
    @test_throws ArgumentError thermal_cft(O, [0.1], [0.1]; virtual_order = :unknown)
    @test_throws ArgumentError thermal_cft(O, [0.1], [0.1]; beta_atol = -1.0)
    @test_throws ArgumentError thermal_cft(O, [0.0, 1e-13], [0.0]; beta_atol = 1e-12)
    @test_throws ArgumentError thermal_cft(ThermalTNR(O; unitcell = (2, 2)), [0.1])
end

@testset "Evolution from beta zero and normalization histories" begin
    for P in (ℂ^2, Z2Space(0 => 1, 1 => 1), U1Space(0 => 1, 1 => 1),
            Vect[FermionParity](0 => 1, 1 => 1))
        O = onsite_pepo(P, 0.1; scalar = ComplexF64)
        original = copy(O)
        grid = collect(0.0:0.1:0.3)
        pivots = [0.3 + 1e-14, 0.0, 0.1]
        rows = thermal_cft(O, grid, pivots; χ_τ = 2, χ_tnr = 2, maxrg = 0, quiet = true)
        @test getproperty.(rows, :depth) == [3, 0, 1]
        @test getproperty.(rows, :beta) ≈ [0.3, 0.0, 0.1]
        @test O == original
        for row in rows
            exact_logz = log(sum(exp.(-row.beta .* [0.3, 1.1])))
            @test row.log_partition_per_site ≈ exact_logz atol = 1e-10
            @test length(row.norms_τ) == row.depth
            @test length(row.cft_history) == length(row.cft_valid) == length(row.norms_xy) == 1
            @test only(row.cft_valid)
            @test abs(only(row.cft_history).central_charge) < 1e-10
        end

        # Including or omitting the explicit beta=0 label gives the same
        # trajectory. Selecting a pivot must not alter subsequent evolution.
        scheme = ThermalTNR(copy(O))
        sparse = run!(scheme, grid[2:end], [0.3]; χ_τ = 2, χ_tnr = 2, maxrg = 0, quiet = true)
        @test only(sparse).log_partition_per_site ≈ first(rows).log_partition_per_site
        expected = onsite_pepo(P, 0.3; scalar = ComplexF64)
        @test pepo_to_2d(scheme.T[1, 1]) ≈ pepo_to_2d(expected / norm(expected))
    end
end

@testset "Fermionic physical trace is the ordinary trace" begin
    P = Vect[FermionParity](0 => 1, 1 => 1)
    O = onsite_pepo(P, 0.0)
    T = pepo_to_2d(O)
    @test norm(T) ≈ 2.0 # The supertrace of this identity would be zero.
    @test [isdual(space(T, i)) for i in 1:4] == [false, false, true, true]
    @test norm(pepo_to_2d(O; is_fermion = false)) < 1e-12
end

@testset "Native TTNR convention and LoopTNR history" begin
    O = classical_ising_3D(0.2)
    original = copy(O)
    wnes = permute(O, ((1, 2), (6, 3, 4, 5)))
    @test pepo_to_2d(O; virtual_order = :NESW) ≈ pepo_to_2d(wnes)
    native = thermal_cft(O, [1, 2], [2];
        virtual_order = :NESW, χ_τ = 4, χ_tnr = 4, maxrg = 1, max_loop = 1, quiet = true)
    adapted = thermal_cft(wnes, [0, 1, 2], [2];
        χ_τ = 4, χ_tnr = 4, maxrg = 1, max_loop = 1, quiet = true)
    @test O == original
    @test only(native).log_partition_per_site ≈ only(adapted).log_partition_per_site atol = 1e-9
    @test length(only(native).cft_history) == 2
    @test getproperty.(only(native).cft_history, :central_charge) ≈
        getproperty.(only(adapted).cft_history, :central_charge) atol = 1e-8
end
