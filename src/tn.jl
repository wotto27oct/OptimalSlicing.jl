struct TensorNetwork
    inputs::Array{Array{Char, 1}, 1}
    output::Array{Char, 1}
    size_dict::Dict{Char, Polynomial}
    parallel_edges::Array{Array{Char, 1}, 1}
end

mutable struct SearchOptions
    max_cost::Union{Polynomial, Nothing}
    max_size::Union{Polynomial, Nothing}
    max_slice::Union{Int, Nothing}
    restrict_outer_product::Bool
    use_cache::Bool
end

function generate_size_dict(inputs)
    size_dict = Dict()
    for input in inputs
        for item in input
            size_dict[item] = Polynomial([0, 1]) # bond dimension chi
        end
    end
    return size_dict
end

function create_3_1_1DTTN()
    inputs = [['b', 'f', 'g', 'h'], ['e', 'f', 'i', 'j'], ['c', 'd', 'i', 'k'], ['j', 'g', 'h', 'l'], ['a', 'b', 'k', 'l']]
    output = ['a', 'c', 'd', 'e']
    parallel_edges = [['c', 'd'], ['g', 'h']]
    max_cost = Polynomial([0, 0, 0, 0, 0, 0, 4])
    max_size = Polynomial([0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, generate_size_dict(inputs), parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_TEBD()
    inputs = [['a', 'b'], ['b', 'c', 'g'], ['c', 'd'], ['d', 'e', 'h'], ['e', 'f'], ['g', 'h', 'i', 'j']]
    output = ['a', 'f', 'i', 'j']
    size_dict = generate_size_dict(inputs)
    size_dict['g'] = size_dict['h'] = size_dict['i'] = size_dict['j'] = Polynomial([2])
    parallel_edges = [['i', 'j']]
    max_cost = Polynomial([0, 0, 16, 10])
    max_size = Polynomial([0, 0, 4])
    tn = TensorNetwork(inputs, output, size_dict, parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_3_1_1DMERA()
    inputs = [['a', 'c', 'd', 'e'], ['b', 'f', 'g', 'h'], ['e', 'f', 'i', 'j'], ['d', 'i', 'k', 'l'], ['l', 'j', 'm', 'n'], ['c', 'k', 'm', 'o'], ['n', 'g', 'h', 'p']]
    output = ['a', 'b', 'o', 'p']
    parallel_edges = [['g', 'h']]
    max_cost = Polynomial([0, 0, 0, 0, 0, 0, 2, 2, 2])
    max_size = Polynomial([0, 0, 0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, generate_size_dict(inputs), parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_9_1_2DTTN()
    inputs = [[0,4,5,6,7,8,9,10,11,12],[1,13,14,15,16,17,18,19,20,21],[3,22,23,24,25,26,27,28,29,30],[12,19,31,22,32,33,34,35],
            [4,5,6,7,8,9,10,11,32,36],[13,14,15,16,17,18,33,20,21,37],[38,39,34,40,41,42,43,44,45,46],[35,23,24,25,26,27,28,29,30,47],[0,1,2,3,36,37,46,47]]
    #[['a', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'], ['b', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v'], 
    # ['d', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E'], ['m', 't', 'F', 'w', 'G', 'H', 'I', 'J'], 
    # ['e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'K'], ['n', 'o', 'p', 'q', 'r', 's', 'H', 'u', 'v', 'L'], 
    # ['M', 'N', 'I', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'], ['J', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'V'], ['a', 'b', 'c', 'd', 'K', 'L', 'U', 'V']]
    output = [2,38,39,31,40,41,42,43,44,45]
    for i in 1:length(inputs)
        for j in 1:length(inputs[i])
            inputs[i][j] = inputs[i][j] + Int('a')
        end
    end
    inputs = [[Char(inputs[i][j]) for j in 1:length(inputs[i])] for i in 1:length(inputs)]
    for i in 1:length(output)
        output[i] = output[i] + Int('a')
    end
    output = [Char(i) for i in output]
    parallel_edges = [['e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'], ['n', 'o', 'p', 'q', 'r', 's', 'u', 'v'],
                    ['x', 'y', 'z', 'A', 'B', 'C', 'D', 'E'], ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']]
    max_cost = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4])
    max_size = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, generate_size_dict(inputs), parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_2DMERA()
    inputs = [['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], ['l', 'q', 'r', 's', 't', 'u', 'v', 'w'], ['x', 'y', 'j', 't', 'z', 'A'], ['B', 'k', 'C', 'v', 'D', 'E'], ['u', 'F', 'G', 'H', 'I', 'J'], ['w', 'K', 'L', 'M', 'N', 'O'], ['P', 'Q', 'i', 'A', 'D', 'R'], ['S', 'z', 'T', 'U', 'I', 'V'], ['E', 'W', 'X', 'O', 'Y', 'Z'], ['J', 'N', 'À', 'Á', 'Â', 'Ã'], ['p', 'q', 'r', 's', 'Ä', 'Å', 'Æ', 'Ç'], ['x', 'y', 'n', 'Ä', 'È', 'É'], ['B', 'o', 'C', 'Æ', 'Ê', 'Ë'], ['Å', 'F', 'G', 'H', 'Ì', 'Í'], ['Ç', 'K', 'L', 'M', 'Î', 'Ï'], ['P', 'Q', 'm', 'É', 'Ê', 'Ð'], ['S', 'È', 'T', 'U', 'Ì', 'Ñ'], ['Ë', 'W', 'X', 'Ï', 'Y', 'Ò'], ['Í', 'Î', 'À', 'Á', 'Â', 'Ó']]
    output = ['R', 'V', 'Z', 'Ã', 'Ð', 'Ñ', 'Ò', 'Ó']
    parallel_edges = []
    max_cost = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 1, 1, 3, 0, 3])
    max_size = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, generate_size_dict(inputs), parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_PEPS(height::Int, width::Int)
    inputs = []
    buff = (height - 1) * width
    for h in 0:height-1
        for w in 0:width-1
            index = [] # up, right, down, left
            if h != 0
                push!(index, (h-1)+w*(height-1))
            end
            if w != width-1
                push!(index, buff+w+h*(width-1))
            end
            if h != height-1
                push!(index, h+w*(height-1))
            end
            if w != 0
                push!(index, buff+(w-1)+h*(width-1))
            end
            push!(inputs, [Char(i+Int('a')) for i in index])
        end
    end
    output = []
    return inputs, output, generate_size_dict(inputs)
end

function create_3_3_PEPS()
    inputs, output, size_dict = create_PEPS(3, 3)
    parallel_edges = []
    max_cost = Polynomial([0, 0, 2, 0, 4, 2, 1])
    max_size = Polynomial([0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, size_dict, parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_periodic_PEPS(height::Int, width::Int)
    inputs = []
    buff = height * width
    for h in 0:height-1
        for w in 0:width-1
            index = [] # up, right, down, left
            push!(index, ((h-1+height)%height)+w*height)
            push!(index, buff+w+h*width)
            push!(index, h+w*height)
            push!(index, buff+((w-1+width)%width)+h*width)
            push!(inputs, [Char(i+Int('a')) for i in index])
        end
    end
    output = []
    return inputs, output, generate_size_dict(inputs)
end

function create_3_3_periodic_PEPS()
    inputs, output, size_dict = create_periodic_PEPS(3, 3)
    parallel_edges = []
    max_cost = Polynomial([0, 0, 0, 0, 1, 0, 0, 3, 3, 1])
    max_size = Polynomial([0, 0, 0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, size_dict, parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_2DHOTRG()
    inputs = [['a', 'b', 'c', 'd'], ['b', 'e', 'f', 'g'], ['c', 'f', 'h'], ['d', 'g', 'i']]
    output = ['a', 'e', 'h', 'i']
    parallel_edges = [['a', 'e'], ['h', 'i']]
    max_cost = Polynomial([0, 0, 0, 0, 0, 0, 2, 1])
    max_size = Polynomial([0, 0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, generate_size_dict(inputs), parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_3DHOTRG()
    inputs = [['a', 'b', 'c', 'd', 'e', 'f'], ['b', 'g', 'h', 'i', 'j', 'k'], ['c', 'h', 'l'], ['d', 'i', 'm'], ['e', 'j', 'n'], ['f', 'k', 'o']]
    output = ['a', 'g', 'l', 'm', 'n', 'o']
    max_cost = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1])
    #parallel_edges = [['a', 'g'], ['l', 'm', 'n', 'o']]
    parallel_edges = [['a', 'g'], ['c', 'd', 'e', 'f']]
    max_size = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, generate_size_dict(inputs), parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_4DHOTRG()
    inputs = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['b', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ['c', 'j', 'p'], ['d', 'k', 'q'], ['e', 'l', 'r'], ['f', 'm', 's'], ['g', 'n', 't'], ['h', 'o', 'u']]
    output = ['a', 'i', 'p', 'q', 'r', 's', 't', 'u']
    parallel_edges = [['a', 'i'], ['c', 'd', 'e', 'f', 'g', 'h']]
    max_cost = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1])
    max_size = Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    tn = TensorNetwork(inputs, output, generate_size_dict(inputs), parallel_edges)
    search_config = SearchOptions(max_cost, max_size, 0, false, false)
    return tn, search_config
end

function create_TN(name::String)
    if name == "3_1_1DTTN"
        return create_3_1_1DTTN()
    elseif name == "TEBD"
        return create_TEBD()
    elseif name == "3_1_1DMERA"
        return create_3_1_1DMERA()
    elseif name == "9_1_2DTTN"
        return create_9_1_2DTTN()
    elseif name == "2DMERA"
        return create_2DMERA()
    elseif name == "3_3_PEPS"
        return create_3_3_PEPS()
    elseif name == "3_3_periodic_PEPS"
        return create_3_3_periodic_PEPS()
    elseif name == "2DHOTRG"
        return create_2DHOTRG()
    elseif name == "3DHOTRG"
        return create_3DHOTRG()
    elseif name == "4DHOTRG"
        return create_4DHOTRG()
    else
        throw(ArgumentError("Unsupported tensor network name: $name"))
    end
end