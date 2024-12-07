using MeanFieldTheory
using QuantumLattices: Bond, BrillouinZone, Fock, Hamiltonian, Hilbert, Hopping, Lattice, MatrixCoupling, Onsite, 𝕗, reciprocals, @σ_str
using TightBindingApproximation: Fermionic, TBA

@time @testset "MeanFieldTheory" begin
    lattice = Lattice([0.0, 0.0], [1.0, 0.0]; vectors=[[1.0, 1.0], [1.0, -1.0]])
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))

    U = 3.0

    t = Hopping(:t, -1.0, 1)
    m = Onsite(:m, 1.0, bond::Bond->iseven(bond[1].site) ? MatrixCoupling(:, 𝕗, :, σ"z", :) : MatrixCoupling(:, 𝕗, :, -σ"z", :))
    λ = Onsite(:λ, 0.0)
    energy(U, m, λ) = m^2/U - 2*λ
    
    free = TBA(lattice, hilbert, t).H
    meanfields = TBA(lattice, hilbert, (m, λ)).H
    ham = Hamiltonian(free, (U=U,), meanfields, energy)
    brillouinzone = BrillouinZone(reciprocals(lattice), 100)
    scmf = SCMF{Fermionic{:TBA}}(lattice, brillouinzone, ham, 0.0)
end