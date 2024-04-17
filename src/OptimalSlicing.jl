module OptimalSlicing

using Polynomials
using Combinatorics
using Format
using Serialization, SHA

include("tn.jl")
export create_TN
include("helpers.jl")
include("search.jl")
export search

end