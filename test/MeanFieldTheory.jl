using MeanFieldTheory
using QuantumLattices: Algorithm, Bond, BrillouinZone, Fock, Hilbert, Hopping, Lattice, Onsite, Parameters, 𝕔⁺𝕔, dimension, reciprocals, scalartype, update!, @σ_str
using TightBindingApproximation: TBA

@time @testset "MeanFieldTheory" begin
    lattice = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))

    T = 0.0
    U = 3.0
    t = Hopping(:t, -1.0, 1)
    m = Onsite(:m, 0.2, bond::Bond->iseven(first(bond).site) ? 𝕔⁺𝕔(:, :, σ"z", :) : 𝕔⁺𝕔(:, :, -σ"z", :))

    afm_normal = TBA(lattice, hilbert, t)
    afm_interaction = (U=U,)
    afm_meanfield = TBA(lattice, hilbert, m)
    afm_constant = (U, m)->m^2/U

    scmf = Algorithm(:HoneycombAFM, SCMF(T, deepcopy(afm_normal), afm_interaction, deepcopy(afm_meanfield), afm_constant, BrillouinZone(reciprocals(lattice), 100)))
    @test scalartype(scmf.frontend) == scalartype(typeof(scmf.frontend)) == Float64
    @test Parameters(scmf.frontend) == (T=T, t=-1.0, U=U, m=0.2)
    @test dimension(scmf.frontend) == 4
    @test normal(scmf) == normal(scmf.frontend) == afm_normal
    @test meanfield(scmf) == meanfield(scmf.frontend) == afm_meanfield
    @test constant(scmf) == constant(scmf.frontend) == 0.2^2/3
    @test Ω(scmf) ≈ Ω(scmf.frontend) ≈ -3.169759127781753
    @test Ω₀(scmf) ≈ Ω₀(scmf.frontend) ≈ -3.1491956389539215
    _, op = optimize!(scmf; condensation=true)
    @test op.minimum ≈ -0.8006432056638455
    @test op.minimizer ≈ [2.5343897118367074]

    scmf = Algorithm(:HoneycombAFM, SCMF(T, deepcopy(afm_normal), afm_interaction, deepcopy(afm_meanfield), afm_constant, BrillouinZone(reciprocals(lattice), Inf)))
    @test Ω(scmf) ≈ Ω(scmf.frontend) ≈ -3.1697591635049887
    @test Ω₀(scmf) ≈ Ω₀(scmf.frontend) ≈ -3.149194521076783
    _, op = optimize!(scmf; condensation=true)
    @test op.minimum ≈ -0.8006443303090753
    @test op.minimizer ≈ [2.5343897176328776]

    update!(scmf; T=1.3)
    _, op = optimize!(scmf; condensation=true)
    @test op.minimum ≈ -0.0016329106484294087
    @test op.minimizer ≈ [0.6975124223291923]
end