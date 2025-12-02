using PolyOCP
using Documenter

DocMeta.setdocmeta!(PolyOCP, :DocTestSetup, :(using PolyOCP); recursive=true)

makedocs(;
    modules=[PolyOCP],
    authors="Ruchuan Ou <ruchuan.ou@tuhh.de> and contributors",
    sitename="PolyOCP.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/OptCon/PolyOCP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/OptCon/PolyOCP.jl",
    devbranch="main",
)
