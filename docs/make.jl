using MeanFieldTheory
using Documenter

DocMeta.setdocmeta!(MeanFieldTheory, :DocTestSetup, :(using MeanFieldTheory); recursive=true)

makedocs(;
    modules=[MeanFieldTheory],
    authors="waltergu <waltergu1989@gmail.com> and contributors",
    sitename="MeanFieldTheory.jl",
    format=Documenter.HTML(;
        canonical="https://Quantum-Many-Body.github.io/MeanFieldTheory.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/MeanFieldTheory.jl",
    devbranch="main",
)
