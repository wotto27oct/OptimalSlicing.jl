"""
    get_index(inputs, T) -> Set

Get the set of indices of the intermediate tensor T.
"""
function get_index(inputs::Array{Array{Char, 1}, 1}, T::Set)
    res = Set()
    for t in T
        res = symdiff(res, Set(inputs[t]))
    end
    return res
end

"""
    get_all_index(inputs, T) -> Set

Get the set of all indices that appear in the contraction of the intermediate tensor T.
"""
function get_all_index(inputs::Array{Array{Char, 1}, 1}, T::Set)
    res = Set()
    for t in T
        res = union(res, Set(inputs[t]))
    end
    return res
end

"""
    get_bdim(idx, size_dict) -> Polynomial

Get the bond dimension of the idx.
"""
function get_bdim(idx::Set, size_dict::Dict{Char, Polynomial})
    res = Polynomial([1])
    for i in idx
        res *= size_dict[i]
    end
    return res
end

"""
    compare_cost(cost1, cost2) -> Int

Compare the cost1 and cost2, which are polynomials.
If cost1 < cost2, then return 1. If cost1 == cost2, then return 0. If cost1 > cost2, then return -1.
"""
function compare_cost(cost1::Polynomial, cost2::Polynomial)
    len1 = degree(cost1) + 1
    len2 = degree(cost2) + 1
    if (len1 < len2) || (len1 == len2 && reverse(coeffs(cost1)) < reverse(coeffs(cost2)))
        return 1
    elseif cost1 == cost2
        return 0
    else
        return -1
    end
end

"""
    compare_cost_int(cost1, cost2) -> Int

Compare the cost1 and cost2, which are integers.
If cost1 < cost2, then return 1. If cost1 == cost2, then return 0. If cost1 > cost2, then return -1.
"""
function compare_cost_int(cost1::Int, cost2::Int)
    if cost1 < cost2
        return 1
    elseif cost1 == cost2
        return 0
    else
        return -1
    end
end

"""
    check_parallel_edges(parallel_edges, inds, sliced_inds) -> Bool

Check if the sliced_inds are selected in order of parallel_edges.
"""
function check_parallel_edges(parallel_edges::Union{Array{Array{Char, 1}, 1}, Nothing}, inds::Set, sliced_inds::Set)
    if isnothing(parallel_edges)
        return true
    end

    for edges in parallel_edges
        # if inds are not subset of parallel edges, then continue
        if !issubset(Set(edges), inds)
            continue
        end
        sliced_edges = []
        for s in sliced_inds
            if s in edges
                push!(sliced_edges, s)
            end
        end
        # if inds are not selected in order of edges, then return false
        if Set(edges[1:length(sliced_edges)]) != Set(sliced_edges)
            return false
        end
    end
    return true
end