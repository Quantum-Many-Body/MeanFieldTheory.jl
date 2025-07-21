module MeanFieldTheory

using Distributed: @distributed
using HCubature: hcubature
using LinearAlgebra: eigvals
using Optim: LBFGS, Options, optimize
using QuantumLattices: Algorithm, BrillouinZone, expand, iscontinuous, isdiscrete, kind, periods, update
using StaticArrays: SVector
using TightBindingApproximation: CompositeTBA, Fermionic, SimpleTBA, TBA, TBAKind

import QuantumLattices: Parameters, dimension, matrix, scalartype, update!
import TightBindingApproximation.Fitting: optimize!

export SCMF, Ω, Ω₀, Ω₀!, constant, normal, meanfield, optimize!

"""
    const PureTBA{K<:TBAKind} = Union{SimpleTBA{K}, CompositeTBA{K}}

Pure tight-binding-approximation, type alias for `Union{SimpleTBA{K}, CompositeTBA{K}}`.
"""
const PureTBA{K<:TBAKind} = Union{SimpleTBA{K}, CompositeTBA{K}}

"""
    SCMF{K<:Fermionic, N<:PureTBA{K}, B<:BrillouinZone, I<:Parameters, M<:PureTBA{K}, C<:Function} <: TBA{K, N, Nothing}

Self-consistent mean-field theory for fermionic systems.
"""
mutable struct SCMF{K<:Fermionic, N<:PureTBA{K}, B<:BrillouinZone, I<:Parameters, M<:PureTBA{K}, C<:Function} <: TBA{K, N, Nothing}
    T::Float64
    const normal::N
    interactions::I
    const meanfield::M
    const constant::C
    const brillouinzone::B
    function SCMF(T::Real, normal::PureTBA, interactions::Parameters, meanfield::PureTBA, constant::Function, brillouinzone::BrillouinZone)
        @assert kind(normal)==kind(meanfield) && normal.lattice==meanfield.lattice && dimension(normal)==dimension(meanfield) "SCMF error: mismatched normal part and mean-field part."
        @assert iscontinuous(brillouinzone) || isdiscrete(brillouinzone) "SCMF error: the input brillouin zone is neither continuous nor discrete."
        new{typeof(kind(normal)), typeof(normal), typeof(brillouinzone), typeof(interactions), typeof(meanfield), typeof(constant)}(T, normal, interactions, meanfield, constant, brillouinzone)
    end
end
@inline scalartype(::Type{<:SCMF{<:Fermionic, N, M}}) where {N<:PureTBA, M<:PureTBA} = promote_type(scalartype(N), scalartype(M))
@inline Parameters(scmf::SCMF) = (; T=scmf.T, Parameters(scmf.normal)..., scmf.interactions..., Parameters(scmf.meanfield)...)
@inline function update!(scmf::SCMF; parameters...)
    if length(parameters)>0
        scmf.T = get(parameters, :T, scmf.T)
        update!(scmf.normal; parameters...)
        scmf.interactions = update(scmf.interactions; parameters...)
        update!(scmf.meanfield; parameters...)
    end
    return scmf
end
@inline dimension(scmf::SCMF) = dimension(scmf.normal)
@inline function matrix(scmf::SCMF, k::Union{AbstractVector{<:Number}, Nothing}=nothing; kwargs...)
    m₁ = matrix(scmf.normal, k; kwargs...)
    m₂ = matrix(scmf.meanfield, k; kwargs...)
    return m₁ + m₂
end

"""
    normal(scmf::SCMF) -> TBA
    normal(scmf::Algorithm{<:SCMF}) -> TBA

Get the normal part.
"""
@inline normal(scmf::SCMF) = scmf.normal
@inline normal(scmf::Algorithm{<:SCMF}) = normal(scmf.frontend)

"""
    meanfield(scmf::SCMF) -> TBA
    meanfield(scmf::Algorithm{<:SCMF}) -> TBA

Get the mean-field part.
"""
@inline meanfield(scmf::SCMF) = scmf.meanfield
@inline meanfield(scmf::Algorithm{<:SCMF}) = meanfield(scmf.frontend)

"""
    constant(scmf::SCMF) -> Real
    constant(scmf::Algorithm{<:SCMF}) -> Real

Get the constant part of the free energy of a fermionic system at the mean-field level.
"""
@inline constant(scmf::Algorithm{<:SCMF}) = constant(scmf.frontend)
@inline constant(scmf::SCMF) = scmf.constant(values(scmf.interactions)..., values(Parameters(scmf.meanfield))...)

"""
    Ω(scmf::SCMF; kwargs...) -> Real
    Ω(scmf::Algorithm{<:SCMF}; kwargs...) -> Real

Get the free energy of a fermionic system at the mean-field level.
"""
@inline Ω(scmf::Algorithm{<:SCMF}; kwargs...) = Ω(scmf.frontend; kwargs...)
@inline Ω(scmf::SCMF; kwargs...) = Ω_(Val(iscontinuous(scmf.brillouinzone)), scmf; kwargs...)
function Ω_(::Val{false}, scmf::SCMF; kwargs...)
    result = (@distributed (+) for k in scmf.brillouinzone
        temp = 0.0
        for e in eigvals(scmf, k; kwargs...)
            temp += Ω_(e, scmf.T; kwargs...)
        end
        temp
    end)::Float64
    result /= length(scmf.brillouinzone)
    result += constant(scmf)
    return result
end
function Ω_(::Val{true}, scmf::SCMF; atol=10^-6, rtol=10^-6, kwargs...)
    function integrand(k)
        result = 0.0
        for e in eigvals(scmf, expand(scmf.brillouinzone, k); kwargs...)
            result += Ω_(e, scmf.T; kwargs...)
        end
        return result
    end
    ps = periods(keytype(scmf.brillouinzone))
    lower, upper = map(p->0.0, ps), map(p->1.0, ps)
    result = hcubature(integrand, lower, upper; atol=atol, rtol=rtol)
    return first(result) + constant(scmf)
end
function Ω_(e::Real, T::Real; cut::Real=100.0, kwargs...)
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

"""
    Ω₀(scmf::SCMF; kwargs...) -> Real
    Ω₀(scmf::Algorithm{<:SCMF}; kwargs...) -> Real

Get the free energy of a fermionic system of the normal state at the mean-field level.
"""
@inline Ω₀(scmf::Algorithm{<:SCMF}; kwargs...) = Ω₀(scmf.frontend; kwargs...)
function Ω₀(scmf::SCMF; kwargs...)
    fields = Parameters(meanfield(scmf))
    params = Parameters{keys(fields)}(map(v->zero(v), values(fields))...)
    return Ω(update!(deepcopy(scmf); params...); kwargs...)
end

"""
    optimize!(
        scmf::Union{SCMF, Algorithm{<:SCMF}}, variables=keys(Parameters(meanfield(scmf)));
        verbose=false,
        method=LBFGS(),
        options=Options(x_abstol=4*10^-6, f_abstol=4*10^-6, iterations=1000, show_trace=true),
        condensation::Bool=false,
        kwargs...
    )

Optimize the order parameters of a fermionic system by the self-consistent mean-field theory.
"""
function optimize!(
    scmf::Union{SCMF, Algorithm{<:SCMF}}, variables=keys(Parameters(meanfield(scmf)));
    verbose=false,
    method=LBFGS(),
    options=Options(x_abstol=4*10^-6, f_abstol=4*10^-6, iterations=1000, show_trace=true),
    condensation::Bool=false,
    kwargs...
)
    v₀ = collect(getfield(Parameters(scmf), name) for name in variables)
    normal = condensation ? Ω₀(scmf; kwargs...) : 0.0
    function objective(v::Vector)
        parameters = Parameters{variables}(v...)
        update!(scmf; parameters...)
        verbose && println(parameters)
        return Ω(scmf; kwargs...) - normal
    end
    v₀ = collect(getfield(Parameters(scmf), name) for name in variables)
    op = optimize(objective, v₀, method, options)
    parameters = Parameters{variables}(op.minimizer...)
    update!(scmf; parameters...)
    return scmf, op
end

end
