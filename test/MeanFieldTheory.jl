using MeanFieldTheory
using QuantumLattices: Bond, BrillouinZone, Fock, Hilbert, Hopping, Lattice, Onsite, 𝕔⁺𝕔, reciprocals, @σ_str
using TightBindingApproximation: Fermionic, TBA

@time @testset "MeanFieldTheory" begin

    lattice = Lattice([0.0, 0.0], [1.0, 0.0]; vectors=[[1.0, 1.0], [1.0, -1.0]])
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))

    U = 3.0
    t = Hopping(:t, -1.0, 1)
    m = Onsite(:m, 1.0, bond::Bond->iseven(first(bond).site) ? 𝕔⁺𝕔(:, :, σ"z", :) : 𝕔⁺𝕔(:, :, -σ"z", :))
    constant(U, m) = m^2/U

    # λ = Onsite(:λ, 0.0)
    # constant(U, m, λ) = m^2/U - 2*λ

    free = TBA(lattice, hilbert, t)
    meanfields = TBA(lattice, hilbert, m)
    brillouinzone = BrillouinZone(reciprocals(lattice), 100)
    scmf = SCMF(0.0, free, (U=U,), meanfields, constant, brillouinzone)
end