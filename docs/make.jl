using OptimalSlicing
using Documenter

DocMeta.setdocmeta!(OptimalSlicing, :DocTestSetup, :(using OptimalSlicing); recursive=true)

makedocs(;
    modules=[OptimalSlicing],
    authors="Hidetaka <u687502j@ecs.osaka-u.ac.jp> and contributors",
    sitename="OptimalSlicing.jl",
    format=Documenter.HTML(;
        canonical="https://wotto27oct.github.io/OptimalSlicing.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/wotto27oct/OptimalSlicing.jl",
    devbranch="main",
)
