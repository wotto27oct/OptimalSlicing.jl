struct Table
    sliced_inds::Dict{Set, Set}
    past_sliced_inds::Dict{Set, Dict}
    costs::Dict{Set, Dict}
    paths::Dict{Set, Dict}
    max_sizes::Dict{Set, Dict}
end

function save_table_cache(tn::TensorNetwork, config::SearchOptions, table::Table, c::Int)
    if config.directory !== nothing
        hash_val = generate_hash(tn, config)
        if !isdir(config.directory)
            mkdir(config.directory)
        end
        cache_path = joinpath(config.directory, "cache_" * hash_val * "_$(c)")
        open(cache_path, "w") do file
            serialize(file, table)
        end
    end
end

function load_table_cache(tn::TensorNetwork, config::SearchOptions)::Tuple{Union{Table, Nothing}, Union{Int, Nothing}}
    if config.directory !== nothing && isdir(config.directory)
        # get the cached table that have the largest number c
        files = readdir(config.directory)

        hash_val = generate_hash(tn, config)
        pattern = "cache_$(hash_val)_(\\d+)"
        max_c = -1

        for file in files
            m = match(Regex(pattern), file)
            if m != nothing
                c_val = parse(Int, m.captures[1])
                if c_val > max_c
                    max_c = c_val
                end
            end
        end

        if max_c >= 0
            cache_path = joinpath(config.directory, "cache_" * hash_val * "_$(max_c)")
            open(cache_path, "r") do file
                table_cache = deserialize(file)
                return (table_cache, max_c)
            end
        else
            return (nothing, nothing)
        end
    else 
        return (nothing, nothing)
    end
end

"""
    get_slicing_cost(tn, config, table, Ta, Tb, Ia, Ib) -> Tuple

Get the cost of the intermediate tensors Ta and Tb with the sliced indices Ia and Ib.
"""
function get_slicing_cost(tn::TensorNetwork, config::SearchOptions, table::Table, Ta::Set, Tb::Set, Ia::Set, Ib::Set)
    # calculate each set of indices of Ta, Tb and Tab
    idxa = get_index(tn.inputs, Ta)
    idxb = get_index(tn.inputs, Tb)
    idx_inner = intersect(idxa, idxb)
    idxa_outer = setdiff(idxa, idx_inner)
    idxb_outer = setdiff(idxb, idx_inner)
    idx_outer = symdiff(idxa, idxb)
    idx_union = union(idxa, idxb)

    # find inner sliced indices of Ia and Ib
    Ia_inner = intersect(Ia, idx_inner)
    Ib_inner = intersect(Ib, idx_inner)
    # If inner sliced indices do not agree, return nothing
    if Ia_inner != Ib_inner
        return (nothing, nothing, nothing)
    end

    # find outer sliced indices of Ia and Ib
    Ia_outer = intersect(Ia, idxa_outer)
    Ib_outer = intersect(Ib, idxb_outer)
    total_idx_outer = setdiff(idx_outer, union(Ia_outer, Ib_outer))

    # find past sliced indices of Ia and Ib
    Ia_past = setdiff(Ia, union(Ia_inner, Ia_outer))
    Ib_past = setdiff(Ib, union(Ib_inner, Ib_outer))
    
    # get the contraction cost of Tab
    cost = get_bdim(idx_union, tn.size_dict)
    if cost == 1
        cost = 0
    end

    # get the size of the sliced intermediate tensor
    intermediate_size = get_bdim(total_idx_outer, tn.size_dict)
    if isnothing(config.max_size) || compare_cost(intermediate_size, config.max_size) >= 0
        if !config.use_cache
            # calculate the slicing overhead in case of using no cache
            slicing_overheada = get_bdim(union(Ib_outer, Ib_past), tn.size_dict)
            slicing_overheadb = get_bdim(union(Ia_outer, Ia_past), tn.size_dict)
            slicing_overhead_past = get_bdim(union(Ia_past, Ib_past), tn.size_dict)
            total_cost = slicing_overhead_past * cost + slicing_overheada * table.costs[Ta][Ia] + slicing_overheadb * table.costs[Tb][Ib]
            # if the total cost exceeds the max_cost, then return nothing
            if !isnothing(config.max_cost) && compare_cost(config.max_cost, total_cost) == 1
                return (nothing, nothing, nothing)
            end
            return (total_cost, intermediate_size, union(Ia, Ib))
        else
            # calculate the slicing overhead in case of using cache
            slicing_overheada = get_bdim(Ib_outer, tn.size_dict)
            slicing_overheadb = get_bdim(Ia_outer, tn.size_dict)
            total_cost = cost + slicing_overheada * table.costs[Ta][Ia] + slicing_overheadb * table.costs[Tb][Ib]
            # if the total cost exceeds the max_cost, then return nothing
            if !isnothing(max_cost) && compare_cost(max_cost, total_cost) == 1
                return (nothing, nothing, nothing)
            end
            return (total_cost, intermediate_size, symdiff(Ia, Ib))
        end
    else
        return (nothing, nothing, nothing)
    end
end

"""
    initialize_table(tn, config, table)

Initialize the table for the bfs.
"""
function initialize_table(tn::TensorNetwork, config::SearchOptions, table::Table)
    N = length(tn.inputs)
    # initialization
    for i in 1:N
        key = Set([i])
        table.sliced_inds[key], table.past_sliced_inds[key], table.costs[key], table.paths[key], table.max_sizes[key] = Set(), Dict(), Dict(), Dict(), Dict()

        inds = get_index(tn.inputs, key)

        # iterate over the number of sliced indices
        for r in 0:length(inds)
            # if r exceeds the max_slice, then break
            if !isnothing(config.max_slice) && r > config.max_slice
                break
            end
            # iterate over the sliced indices with size r
            for s in combinations(collect(inds), r)
                s_set = Set(s)
                # if the initial size of tensor already exceeds the max_size, then skip
                if !isnothing(config.max_size) && compare_cost(config.max_size, get_bdim(setdiff(Set(inds), s_set), tn.size_dict)) == 1
                    continue
                end
                # check the parallel edges
                if !check_parallel_edges(tn.parallel_edges, inds, s_set)
                    continue
                end
                push!(table.sliced_inds[key], Set(s))
                table.past_sliced_inds[key][Set(s)] = s_set
                table.costs[key][Set(s)] = 0
                table.paths[key][Set(s)] = (i,)
                table.max_sizes[key][Set(s)] = get_bdim(setdiff(Set(inds), s_set), tn.size_dict)
            end
            # if config.maxsize is nothing, then break (we don't care about slicing)
            if r == 0 && isnothing(config.max_size)
                break
            end
        end
    end
end

"""
    update_intermediate_tensor(tn, config, table, Ta, Tb)

Update the table of intermediate tensor Tab given Ta and Tb.
"""
function update_intermediate_tensor(tn::TensorNetwork, config::SearchOptions, table::Table, Ta::Set, Tb::Set)
    Tab = union(Ta, Tb)

    # update intermediate tensors
    for Ia in table.sliced_inds[Ta]
        for Ib in table.sliced_inds[Tb]
            # limit the number of sliced indices
            if !isnothing(config.max_slice) && length(union(table.past_sliced_inds[Ta][Ia], table.past_sliced_inds[Tb][Ib])) > config.max_slice
                continue
            end

            # use the symetric property of the edges
            if !check_parallel_edges(tn.parallel_edges, get_all_index(tn.inputs, Tab), union(table.past_sliced_inds[Ta][Ia], table.past_sliced_inds[Tb][Ib]))
                continue
            end
            
            # get the cost of the intermediate tensor
            new_cost, new_size, Iab = get_slicing_cost(tn, config, table, Ta, Tb, Ia, Ib)
            if isnothing(new_cost)
                continue
            end

            # if Tab is not in the table, then initialize it
            if !haskey(table.sliced_inds, Tab)
                table.sliced_inds[Tab] = Set()
                table.past_sliced_inds[Tab] = Dict()
                table.costs[Tab] = Dict()
                table.paths[Tab] = Dict()
                table.max_sizes[Tab] = Dict()
            end

            # calculate the size of contracting Tab
            if compare_cost(new_size, table.max_sizes[Ta][Ia]) == 1
                new_size = table.max_sizes[Ta][Ia]
            end
            if compare_cost(new_size, table.max_sizes[Tb][Ib]) == 1
                new_size = table.max_sizes[Tb][Ib]
            end

            if !(Iab in table.sliced_inds[Tab])
                # if Iab is not in the table, then add it
                push!(table.sliced_inds[Tab], union(table.past_sliced_inds[Ta][Ia], table.past_sliced_inds[Tb][Ib]))
                table.past_sliced_inds[Tab][Iab] = union(table.past_sliced_inds[Ta][Ia], table.past_sliced_inds[Tb][Ib])
                table.costs[Tab][Iab] = new_cost
                table.paths[Tab][Iab] = (table.paths[Ta][Ia], table.paths[Tb][Ib])
                table.max_sizes[Tab][Iab] = new_size
                continue
            else
                # If Iab is in the table, compare cost and size with the existing one and update it.
                cost_compare_result = compare_cost(new_cost, table.costs[Tab][Iab])
                size_compare_result = compare_cost(new_size, table.max_sizes[Tab][Iab])
                slice_compare_result = compare_cost_int(length(union(table.past_sliced_inds[Ta][Ia], table.past_sliced_inds[Tb][Ib])), length(table.past_sliced_inds[Tab][Iab]))
                # update if
                # 1. new_cost < table.costs[Tab][Iab]
                # 2. new_cost == table.costs[Tab][Iab] && new_size < table.max_sizes[Tab][Iab]
                # 3. new_cost == table.costs[Tab][Iab] && new_size == table.max_sizes[Tab][Iab] && # of sliced indices is smaller
                if cost_compare_result == 1 || (cost_compare_result == 0 && slice_compare_result == 1) || (cost_compare_result == 0 && slice_compare_result == 0 && size_compare_result == 1)
                    table.past_sliced_inds[Tab][Iab] = union(table.past_sliced_inds[Ta][Ia], table.past_sliced_inds[Tb][Ib])
                    table.costs[Tab][Iab] = new_cost
                    table.paths[Tab][Iab] = (table.paths[Ta][Ia], table.paths[Tb][Ib])
                    table.max_sizes[Tab][Iab] = new_size
                end
            end
        end
    end
end

"""
    update_table(tn, config, table)

Update the table of intermediate tensors.
"""
function update_table(tn::TensorNetwork, config::SearchOptions, table::Table, c_max::Union{Int, Nothing})
    N = length(tn.inputs)
    intermediates = [i == 1 ? Set([Set([i]) for i in 1:N]) : Set() for i in 1:N]

    if !isnothing(c_max)
        for Tab in keys(table.costs)
            push!(intermediates[length(Tab)], Tab)
        end
    end

    # iterate over the number of intermediate tensors
    for c in 2:N
        if !isnothing(c_max) && c <= c_max
            println("load cached table with c=$c")
            continue
        end
        dmax = c ÷ 2
        # contract size_d tensor and size_e tensor to create size_c tensor (d <= e)
        for d in 1:dmax
            e = c - d
            println("create $c tensor using $d and $e")
            pairs = []
            if d < e
                pairs = [(x, y) for x in intermediates[d] for y in intermediates[e]]
            else
                intermediate_d = collect(intermediates[d])
                pairs = [(intermediate_d[i], intermediate_d[j]) for i in 1:length(intermediate_d) for j in i:length(intermediate_d)]
            end
            for (Ta, Tb) in pairs
                # continue if Ta and Tb share the same tensor.
                if length(intersect(Ta, Tb)) > 0
                    continue
                end
                # continue if Ta and Tb have no common index.
                if config.restrict_outer_product && length(intersect(get_index(tn.inputs, Ta), get_index(tn.inputs, Tb))) == 0
                    continue
                end
                # update the table of intermediate tensor Tab
                update_intermediate_tensor(tn, config, table, Ta, Tb)
                Tab = union(Ta, Tb)
                if Tab in keys(table.costs)
                    push!(intermediates[c], Tab)
                end
            end
        end
        num_candidates = 0
        for Tab in keys(table.costs)
            if length(Tab) == c
                if length(table.costs[Tab]) > 0
                    num_candidates += length(table.costs[Tab])
                end
            end
        end
        println("# candidates of size $c: $num_candidates")
        save_table_cache(tn, config, table, c)
    end
end

"""
    nested_to_ssa(nested_path::Tuple, N::Int) -> Array{Tuple{Int, Int}, 1}

Convert the nested path to the ssa (single static assignment) path.

Example:
    nested_to_ssa (((3,), (2,)), ((5,), ((1,), (4,)))) -> [(2, 1), (0, 3), (4, 6), (5, 7)]
"""
function nested_to_ssa(nested_path::Tuple, N::Int)
    ssa_path = Tuple{Int, Int}[]
    function recursive(p::Tuple, node_num::Int)
        if length(p) == 1
            return p[1]
        else
            left = recursive(p[1], node_num)
            right = recursive(p[2], max(left, node_num))
            push!(ssa_path, (left, right))
            if left <= N && right <= N
                return node_num + 1
            else
                return max(left, right) + 1
            end
        end
    end
    recursive(nested_path, N)
    return ssa_path
end

"""
    ssa_to_linear(ssa_path::Array{Tuple{Int, Int}, 1}) -> Array{Tuple{Int, Int}, 1}

Convert the ssa (single static assignment) path to the linear path (conventional contraction path).
"""
function ssa_to_linear(ssa_path::Array{Tuple{Int, Int}, 1})
    max_ssa_id = maximum(maximum.(ssa_path))
    ids = collect(1:max_ssa_id+1)

    path = Tuple{Int, Int}[]
    for ssa_ids in ssa_path
        current_pair = Tuple(ids[ssa_id] for ssa_id in ssa_ids)
        push!(path, current_pair)

        for ssa_id in ssa_ids
            ids[ssa_id:end] .-= 1
        end
    end
    return path
end

function linear_to_ssa(path::Array{Tuple{Int, Int}, 1})
    N = length(path) + 1
    ids = collect(0:N-1)
    ssa = N
    ssa_path = Tuple{Int, Int}[]
    for con in path
        sorted_con = sort(collect(con), rev=true)
        scon = [popat!(ids, c+1) for c in sorted_con]
        push!(ssa_path, tuple(scon...))
        push!(ids, ssa)
        ssa += 1
    end
    return ssa_path
end

"""
    path_to_0_indexed(path::Array{Tuple{Int, Int}, 1}) -> Array{Tuple{Int, Int}, 1}

Convert the path to the 0-indexed path.
"""
function path_to_0_indexed(path::Array{Tuple{Int, Int}, 1})
    return [(a-1, b-1) for (a, b) in path]
end

"""
    get_best_results(tn, config, table) -> Tuple

Get the optimal result of contracting the final tensor from the table.
"""
function get_best_results(tn::TensorNetwork, config::SearchOptions, table::Table)
    N = length(tn.inputs)
    best_cost = nothing
    best_max_size = nothing
    best_path = nothing
    best_slices = nothing

    for key in keys(table.costs[Set(1:N)])
        if !isnothing(best_cost)
            cost_compare_result = compare_cost(table.costs[Set(1:N)][key], best_cost)
        end
        # update if
        # 1. best_cost is nothing
        # 2. table.costs[Set(1:N)][key] < best_cost
        # 3. table.costs[Set(1:N)][key] == best_cost && # of sliced indices is smaller
        if isnothing(best_cost) || cost_compare_result == 1 || (cost_compare_result == 0 && length(best_slices) > length(table.past_sliced_inds[Set(1:N)][key]))
            best_cost = table.costs[Set(1:N)][key]
            best_path = table.paths[Set(1:N)][key]
            best_max_size = table.max_sizes[Set(1:N)][key]
            best_slices = table.past_sliced_inds[Set(1:N)][key]
        end
    end

    return best_cost, best_max_size, best_path, best_slices
end

struct PathInfo
    cost::Polynomial
    max_size::Polynomial
    path::Array{Tuple{Int, Int}, 1}
    slices::Set{Char}
end

function format_pathinfo(pathinfo::PathInfo)
    path_print = "Results:\n"
    path_print *= "cost: $(pathinfo.cost)\n"
    path_print *= "max_size: $(pathinfo.max_size)\n"
    path_print *= "linear path: $(pathinfo.path)\n"
    path_print *= "ssa path: $(linear_to_ssa(pathinfo.path))\n"
    path_print *= "slices: $(pathinfo.slices)\n"
    return path_print
end

struct LineInfo
    order::String
    left_indices::String
    right_indices::String
    output_indices::String
    blas_cost::String
    slice_cost::String
    blas_size::String
end

function print_path(tn::TensorNetwork, pathinfo::PathInfo)
    path_print = "Contraction Details:\n"
    tensor_indices = copy(tn.inputs)

    function decorate_sliced_inds(indices)
        index_str = ""
        for index in indices
            if index in pathinfo.slices
                index_str *= "{$index}"
            else
                index_str *= string(index)
            end
        end
        return index_str
    end

    line_info_list = LineInfo[]
    push!(line_info_list, LineInfo("", "left", "right", "output", "blas cost", "slicing overhead", "blas size"))

    slice_cost = get_bdim(pathinfo.slices, tn.size_dict)

    for (i, (a, b)) in enumerate(pathinfo.path)
        left_indices = decorate_sliced_inds(tensor_indices[a+1])
        right_indices = decorate_sliced_inds(tensor_indices[b+1])
        output = symdiff(tensor_indices[a+1], tensor_indices[b+1])
        blas_cost = get_bdim(Set(setdiff(union(tensor_indices[a+1], tensor_indices[b+1]), pathinfo.slices)), tn.size_dict)
        blas_size = [get_bdim(Set(setdiff(tensor_indices[a+1], pathinfo.slices)), tn.size_dict), get_bdim(Set(setdiff(tensor_indices[a+1], pathinfo.slices)), tn.size_dict), get_bdim(Set(setdiff(output, pathinfo.slices)), tn.size_dict)]
        output_indices = decorate_sliced_inds(output)
        push!(line_info_list, LineInfo(string(i), left_indices, right_indices, output_indices, string(blas_cost), string(slice_cost), "$(blas_size[1]), $(blas_size[2]) -> $(blas_size[3])"))
        deleteat!(tensor_indices, sort([a+1, b+1]))
        push!(tensor_indices, output)
    end

    function get_max_length(line_info_list::Vector{LineInfo}, field::Symbol)::Int
        max_length = 0
        for line_info in line_info_list
            field_value = getfield(line_info, field)
            max_length = max(max_length, length(field_value))
        end
        return max_length
    end

    lengths = (max(3, get_max_length(line_info_list, :order)+2),
                max(10, get_max_length(line_info_list, :left_indices)+2),
                max(10, get_max_length(line_info_list, :right_indices)+2),
                max(10, get_max_length(line_info_list, :output_indices)+2),
                max(10, get_max_length(line_info_list, :blas_cost)+2),
                max(10, get_max_length(line_info_list, :slice_cost)+2),
                max(10, get_max_length(line_info_list, :blas_size)+2))

    function format_path(line_info)
        fe = FormatExpr(join(["{:<$(lengths[i])}" for i in 1:length(lengths)], " ") * "\n")
        formatted_string = format(fe, line_info.order, line_info.left_indices, line_info.right_indices, line_info.output_indices, line_info.blas_cost, line_info.slice_cost, line_info.blas_size)
        return formatted_string
    end

    for line_info in line_info_list
        path_print *= format_path(line_info)
    end
    return path_print
end

function print_contraction(tn::TensorNetwork, config::SearchOptions, pathinfo::PathInfo)
    path_print = ""
    path_print *= "----------------------------------------\n"
    path_print *= format_tensor_network(tn)
    path_print *= "----------------------------------------\n"
    path_print *= format_search_options(config)
    path_print *= "----------------------------------------\n"
    path_print *= format_pathinfo(pathinfo)
    path_print *= "----------------------------------------\n"
    path_print *= print_path(tn, pathinfo)
    return path_print
end

"""
    search(tn, config)

Search the optimal contraction path and slicing of the tensor network.
"""
function search(tn::TensorNetwork, config::SearchOptions)
    N = length(tn.inputs)

    # load cached table
    table, c_max = load_table_cache(tn, config)
    if isnothing(table)
        # define the table
        table = Table(Dict(), Dict(), Dict(), Dict(), Dict())
        # initialize the table
        initialize_table(tn, config, table)
    end

    # update the table
    update_table(tn, config, table, c_max)

    # get the best results
    best_cost, best_max_size,  best_path, best_slices = get_best_results(tn, config, table)
    ssa_path = nested_to_ssa(best_path, N)
    linear_path = ssa_to_linear(ssa_path)
    pathinfo::PathInfo = PathInfo(best_cost, best_max_size, path_to_0_indexed(linear_path), best_slices)
    path_print = print_contraction(tn, config, pathinfo)
    return pathinfo, path_print
end

"""
    search(tn_name::String)

Search the optimal contraction path and slicing of the tensor network with the given name.
"""
function search(tn_name::String)
    tn, config = create_TN(tn_name)
    return search(tn, config)
end