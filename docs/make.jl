using PolyOCP
using Documenter

DocMeta.setdocmeta!(PolyOCP, :DocTestSetup, :(using PolyOCP); recursive=true)

makedocs(;
    modules=[PolyOCP],
    authors="Ruchuan Ou <ruchuan.ou@tuhh.de> and contributors",
    sitename="PolyOCP.jl",
    format=Documenter.HTML(;
        canonical="https://Ruchuan.github.io/PolyOCP.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Ruchuan/PolyOCP.jl",
    devbranch="master",
)
