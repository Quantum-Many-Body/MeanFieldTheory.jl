module MeanFieldTheory

using ChainRulesCore: HasReverseMode, NoTangent, RuleConfig, rrule_via_ad
using Distributed: @distributed
using LinearAlgebra: eigvals
using QuantumLattices: BrillouinZone, kind, update
using TightBindingApproximation: CompositeTBA, Fermionic, SimpleTBA, TBA, TBAKind

import ChainRulesCore: rrule
import QuantumLattices: Parameters, dimension, matrix, scalartype, update!

export SCMF, Ω, constant

const PureTBA{K<:TBAKind} = Union{SimpleTBA{K}, CompositeTBA{K}}

mutable struct SCMF{K<:Fermionic, F<:PureTBA{K}, I<:Parameters, M<:PureTBA{K}, C<:Function, B<:BrillouinZone} <: TBA{K, F, Nothing}
    T::Float64
    const free::F
    interactions::I
    const meanfields::M
    const constant::C
    const brillouinzone::B
    function SCMF(T::Real, free::PureTBA, interactions::Parameters, meanfields::TBA, constant::Function, brillouinzone::BrillouinZone)
        @assert kind(free)==kind(meanfields) && free.lattice==meanfields.lattice && dimension(free)==dimension(meanfields) "SCMF error: mismatched free part and meanfields part."
        new{typeof(kind(free)), typeof(free), typeof(interactions), typeof(meanfields), typeof(constant), typeof(brillouinzone)}(T, free, interactions, meanfields, constant, brillouinzone)
    end
end
@inline scalartype(::Type{<:SCMF{<:Fermionic, F, M}}) where {F<:PureTBA, M<:PureTBA} = promote_type(scalartype(F), scalartype(M))
@inline Parameters(scmf::SCMF) = (; T=scmf.T, Parameters(scmf.free)..., scmf.interactions..., Parameters(scmf.meanfields)...)
@inline function update!(scmf::SCMF; parameters...)
    if length(parameters)>0
        scmf.T = get(parameters, :T, scmf.T)
        update!(scmf.free; parameters...)
        scmf.interactions = update(scmf.interactions; parameters...)
        update!(scmf.meanfields; parameters...)
    end
    return scmf
end
@inline dimension(scmf::SCMF) = dimension(scmf.free)
@inline function matrix(scmf::SCMF, k::Union{AbstractVector{<:Number}, Nothing}=nothing; kwargs...)
    m₁ = matrix(scmf.free, k; kwargs...)
    m₂ = matrix(scmf.meanfields, k; kwargs...)
    return m₁ + m₂
end
@inline function constant(scmf::SCMF)
    return scmf.constant(values(scmf.interactions)..., values(Parameters(scmf.meanfields))...)
end

"""
"""
function (scmf::SCMF)(vs::Number...; kwargs...)
    update!(scmf; Parameters{keys(Parameters(scmf.meanfields))}(vs...)...)
    result = (@distributed (+) for k in scmf.brillouinzone
        temp = 0.0
        for e in eigvals(scmf, k; kwargs...)
            temp += Ω(e, scmf.T; kwargs...)
        end
        temp
    end)::Float64
    result /= length(scmf.brillouinzone)
    result += constant(scmf)
    return result
end

function rrule(config::RuleConfig{>:HasReverseMode}, scmf::SCMF, vs::Number...; kwargs...)
    vs = promote(vs...)
    update!(scmf; Parameters{keys(Parameters(scmf.meanfields))}(vs...)...)
    primal = 0.0
    dvs = zeros(eltype(vs), length(vs))
    Δes = zeros(eltype(vs), length(vs))
    Δms = []
    for k in scmf.brillouinzone
        m = matrix(scmf, k; kwargs...)
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
    c, c_pullback = rrule_via_ad(config, scmf.constant, values(scmf.interactions)..., vs...)
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
