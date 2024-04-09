module OptimalSlicing

using Polynomials
using Combinatorics

include("tn.jl")
export create_TN
include("helpers.jl")
include("search.jl")
export search

end