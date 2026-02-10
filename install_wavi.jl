using Pkg ; Pkg.add("IJulia")
using IJulia ; installkernel("Julia")

# Julia packages used by the IceGraph WAVI notebook.
Pkg.add("Glob") ; Pkg.add("JLD2") ; Pkg.add("Plots")
Pkg.add(PackageSpec(url="https://github.com/WAVI-ice-sheet-model/WAVI.jl.git", rev = "main"))