module MeanFieldTheory

using ChainRulesCore: HasReverseMode, NoTangent, RuleConfig, rrule_via_ad
using Distributed: @distributed
using LinearAlgebra: eigvals
using QuantumLattices: AbstractLattice, BrillouinZone, update
using TightBindingApproximation: AbstractTBA, Fermionic

import ChainRulesCore: rrule
import QuantumLattices: Hamiltonian, Parameters, dimension, matrix, update!

export SCMF, SCMFHamiltonian, Ω, constant

"""
"""
mutable struct SCMFHamiltonian{F<:Hamiltonian, I<:Parameters, M<:Hamiltonian, C<:Function} <: Hamiltonian
    const free::F
    interactions::I
    const meanfields::M
    const constant::C
end
@inline Base.valtype(::Type{<:SCMFHamiltonian{F, <:Parameters, M}}) where {F<:Hamiltonian, M<:Hamiltonian} = promote_type(valtype(F), valtype(M))
@inline Parameters(hamiltonian::SCMFHamiltonian) = (; Parameters(hamiltonian.free)..., hamiltonian.interactions..., Parameters(hamiltonian.meanfields)...)
@inline function update!(hamiltonian::SCMFHamiltonian; k=nothing, parameters...)
    if length(parameters)>0
        update!(hamiltonian.free; parameters...)
        hamiltonian.interactions = update(hamiltonian.interactions; parameters...)
        update!(hamiltonian.meanfields; parameters...)
    end
    return hamiltonian
end
@inline dimension(hamiltonian::SCMFHamiltonian) = dimension(hamiltonian.free)
@inline function matrix(hamiltonian::SCMFHamiltonian; k=nothing, kwargs...)
    m₁ = matrix(hamiltonian.free; k=k, kwargs...)
    m₂ = matrix(hamiltonian.meanfields; k=k, kwargs...)
    return m₁ + m₂
end
@inline function constant(hamiltonian)
    return hamiltonian.constant(values(hamiltonian.interactions)..., values(Parameters(hamiltonian.meanfields))...)
end

"""
"""
@inline function Hamiltonian(free::Hamiltonian, interactions::Parameters, meanfields::Hamiltonian, constant::Function)
    @assert dimension(free)==dimension(meanfields) "Hamiltonian error: mismatched dimension of the free part and the mean-field part."
    return SCMFHamiltonian(free, interactions, meanfields, constant)
end

"""
"""
mutable struct SCMF{K<:Fermionic, L<:AbstractLattice, B<:BrillouinZone, H<:SCMFHamiltonian} <: AbstractTBA{K, H, Nothing}
    const lattice::L
    const brillouinzone::B
    const H::H
    T::Float64
    function SCMF{K}(lattice::AbstractLattice, brillouinzone::BrillouinZone, H::SCMFHamiltonian, T::Real=0.0) where {K<:Fermionic}
        new{K, typeof(lattice), typeof(brillouinzone), typeof(H)}(lattice, brillouinzone, H, T)
    end
end
@inline Parameters(scmf::SCMF) = (; T=scmf.T, Parameters(scmf.H)...)
@inline function update!(scmf::SCMF; kwargs...)
    scmf.T = get(kwargs, :T, scmf.T)
    update!(scmf.H; kwargs...)
    return scmf
end

"""
"""
function (scmf::SCMF)(vs::Number...; kwargs...)
    update!(scmf; Parameters{keys(Parameters(scmf.H.meanfields))}(vs...)...)
    result = @distributed (+) for k in scmf.brillouinzone
        temp = 0.0
        for e in eigvals(scmf; k=k, kwargs...)
            temp += Ω(e, scmf.T; kwargs...)
        end
        temp
    end
    result /= length(scmf.brillouinzone)
    result += constant(scmf.H)
    return result
end
function rrule(config::RuleConfig{>:HasReverseMode}, scmf::SCMF, vs::Number...; kwargs...)
    vs = promote(vs...)
    update!(scmf; Parameters{keys(Parameters(scmf.H.meanfields))}(vs...)...)
    primal = 0.0
    dvs = zeros(eltype(vs), length(vs))
    Δes = zeros(eltype(vs), length(vs))
    Δms = []
    for k in scmf.brillouinzone
        m = matrix(scmf; k=k, kwargs...)
        es, es_pullback = rrule(eigvals, m)
        for (i, e) in enumerate(es)
            ω, ω_pullback = rrule(Ω, e, scmf.T; kwargs...)
            primal += ω
            Δes[i] = ω_pullback(1.0)[2]
        end
        Δm = es_pullback(Δes)
        for i in eachindex(dvs)
            dvs[i] += mapreduce(*, +, Δm, Δms[i])
        end
    end
    primal /= length(scmf.brillouinzone)
    for i in eachindex(dvs)
        dvs[i] /= length(scmf.brillouinzone)
    end
    c, c_pullback = rrule_via_ad(config, scmf.H.constant, values(scmf.H.interactions)..., vs...)
    primal += c
    for (i, dc) in enumerate(c_pullback(1.0)[end-length(vs):end])
        dvs[i] += Δc[index]
    end
    scmf_pullback(Δscmf) = (NoTangent(), ntuple(i->Δscmf*dvs[i], Val(fieldcount(typeof(vs))))...)
    return primal, scmf_pullback
end

"""
"""
function Ω(e::Real, T::Real; cut::Real=100.0, kwargs...)
    e = convert(Float64, e)
    T = convert(Float64, T)
    ratio = e/T
    result = if isnan(ratio)
        zero(e)
    elseif abs(ratio)>cut
        e>0 ? zero(e) : e
    else
        - T * log(1+exp(-ratio))
    end
    return result
end
function rrule(::typeof(Ω), e::Real, T::Real; cut::Real=100.0, kwargs...)
    e = convert(Float64, e)
    T = convert(Float64, T)
    ratio = e/T
    primal, de, dT = if abs(ratio)>cut
        e>0 ? (zero(e), zero(e), zero(e)) : (e, one(e), zero(e))
    else
        x = exp(-ratio)
        y = 1 + x
        z = log(y)
        (-T*z, x/y, -z-(ratio*x)/y)
    end
    Ω_pullback(ΔΩ) = NoTangent(), ΔΩ*de, ΔΩ*dT
    return primal, Ω_pullback
end

end
