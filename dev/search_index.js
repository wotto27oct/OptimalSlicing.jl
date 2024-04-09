var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = OptimalSlicing","category":"page"},{"location":"#OptimalSlicing","page":"Home","title":"OptimalSlicing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for OptimalSlicing.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Optimal Slicing find the optimal contraction path and slicing indices given a tensor network.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [OptimalSlicing]","category":"page"},{"location":"#OptimalSlicing.check_parallel_edges-Tuple{Union{Nothing, Vector{Vector{Char}}}, Set, Set}","page":"Home","title":"OptimalSlicing.check_parallel_edges","text":"check_parallel_edges(parallel_edges, inds, sliced_inds) -> Bool\n\nCheck if the slicedinds are selected in order of paralleledges.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.compare_cost-Tuple{Polynomials.Polynomial, Polynomials.Polynomial}","page":"Home","title":"OptimalSlicing.compare_cost","text":"compare_cost(cost1, cost2) -> Int\n\nCompare the cost1 and cost2, which are polynomials. If cost1 < cost2, then return 1. If cost1 == cost2, then return 0. If cost1 > cost2, then return -1.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.compare_cost_int-Tuple{Int64, Int64}","page":"Home","title":"OptimalSlicing.compare_cost_int","text":"compare_cost_int(cost1, cost2) -> Int\n\nCompare the cost1 and cost2, which are integers. If cost1 < cost2, then return 1. If cost1 == cost2, then return 0. If cost1 > cost2, then return -1.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.get_all_index-Tuple{Vector{Vector{Char}}, Set}","page":"Home","title":"OptimalSlicing.get_all_index","text":"get_all_index(inputs, T) -> Set\n\nGet the set of all indices that appear in the contraction of the intermediate tensor T.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.get_bdim-Tuple{Set, Dict{Char, Polynomials.Polynomial}}","page":"Home","title":"OptimalSlicing.get_bdim","text":"get_bdim(idx, size_dict) -> Polynomial\n\nGet the bond dimension of the idx.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.get_best_results-Tuple{OptimalSlicing.TensorNetwork, OptimalSlicing.SearchOptions, OptimalSlicing.Table}","page":"Home","title":"OptimalSlicing.get_best_results","text":"get_best_results(tn, config, table) -> Tuple\n\nGet the optimal result of contracting the final tensor from the table.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.get_index-Tuple{Vector{Vector{Char}}, Set}","page":"Home","title":"OptimalSlicing.get_index","text":"get_index(inputs, T) -> Set\n\nGet the set of indices of the intermediate tensor T.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.get_slicing_cost-Tuple{OptimalSlicing.TensorNetwork, OptimalSlicing.SearchOptions, OptimalSlicing.Table, Vararg{Set, 4}}","page":"Home","title":"OptimalSlicing.get_slicing_cost","text":"get_slicing_cost(tn, config, table, Ta, Tb, Ia, Ib) -> Tuple\n\nGet the cost of the intermediate tensors Ta and Tb with the sliced indices Ia and Ib.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.initialize_table-Tuple{OptimalSlicing.TensorNetwork, OptimalSlicing.SearchOptions, OptimalSlicing.Table}","page":"Home","title":"OptimalSlicing.initialize_table","text":"initialize_table(tn, config, table)\n\nInitialize the table for the bfs.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.search-Tuple{OptimalSlicing.TensorNetwork, OptimalSlicing.SearchOptions}","page":"Home","title":"OptimalSlicing.search","text":"search(tn, config)\n\nSearch the optimal contraction path and slicing of the tensor network.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.search-Tuple{String}","page":"Home","title":"OptimalSlicing.search","text":"search(tn_name::String)\n\nSearch the optimal contraction path and slicing of the tensor network with the given name.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.update_intermediate_tensor-Tuple{OptimalSlicing.TensorNetwork, OptimalSlicing.SearchOptions, OptimalSlicing.Table, Set, Set}","page":"Home","title":"OptimalSlicing.update_intermediate_tensor","text":"update_intermediate_tensor(tn, config, table, Ta, Tb)\n\nUpdate the table of intermediate tensor Tab given Ta and Tb.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalSlicing.update_table-Tuple{OptimalSlicing.TensorNetwork, OptimalSlicing.SearchOptions, OptimalSlicing.Table}","page":"Home","title":"OptimalSlicing.update_table","text":"update_table(tn, config, table)\n\nUpdate the table of intermediate tensors.\n\n\n\n\n\n","category":"method"}]
}
