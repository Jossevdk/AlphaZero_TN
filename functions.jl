#Edges are ordered when first looping over the list of lists 
#and then over this list only counted once with taking the upper triangle, 
#with the index of a column equal to the index of a list in the list of lists

mutable struct BatchFeatureGraph
    fg::FeaturedGraph
    graph_indicator::Vector{Int64}
    function BatchFeatureGraph(fg::FeaturedGraph, graph_indicator::Vector{Int64})
        new(fg, graph_indicator)
    end
end

function Flux.batch(fgs::AbstractVector{<:FeaturedGraph})
    v_num_nodes = [nv(fg) for fg in fgs]
    nodesum = cumsum([0; v_num_nodes])[1:(end - 1)]
    rows = vcat([findnz(fg.graph.S)[1] .+ nodesum[i] for (i, fg) in enumerate(fgs)]...)
    cols = vcat([findnz(fg.graph.S)[2] .+ nodesum[i] for (i, fg) in enumerate(fgs)]...)
    vals = vcat([findnz(fg.graph.S)[3] for fg in fgs]...)
    # print(rows, cols, vals)
    S = sparse(rows, cols, vals)
    ef = hcat([edge_feature(fg) for fg in fgs]...)
    nf = hcat([node_feature(fg) for fg in fgs]...)
    
    #nf = hcat([fg.nf for fg in fgs]...)

    function materialize_graph_indicator(fg)
        ones(Int64, nv(fg))
    end

    v_gi = materialize_graph_indicator.(fgs)
    
    graphsum = 0:(length(fgs)-1)
    v_gi = [ng .+ gi for (ng, gi) in zip(graphsum, v_gi)]
    graph_indicator = vcat(v_gi...)
    

    return BatchFeatureGraph(FeaturedGraph(S, nf= nf, ef = ef), graph_indicator)
end
function Base.copy(fg::FeaturedGraph)
    return FeaturedGraph(fg)
end

# Define hash function for FeaturedGraph
function Base.hash(fg::FeaturedGraph, h::UInt)
    return hash(fg.graph, hash(fg.ef, hash(fg.nf, h)))
end

function Base.isequal(fg::FeaturedGraph, fh::FeaturedGraph)
    return fg.graph == fh.graph && fg.ef == fh.ef && fg.nf == fh.nf
end

function Base.:(==)(fg::FeaturedGraph, fh::FeaturedGraph)
    return isequal(fg, fh)
end


function Base.:(==)(sg::SparseGraph, sh::SparseGraph)
    return sg.S == sh.S && sg.edges == sh.edges && sg.E == sh.E
end


function get_edges_ary(con_ary::Array{Int64, 2})
    edges_ary = zeros(Int, size(con_ary))
    
    edges_ary[con_ary .== -1] .= -1
  
    num_edges = 1
    for i in axes(con_ary, 1)
        for j in axes(con_ary, 2)
            if con_ary[i, j] == -1
                continue
            end
            if i == con_ary[i, j]
                edges_ary[i, i] = num_edges
                num_edges +=1
            end
            if i>con_ary[i, j]
                edges_ary[i, j] = num_edges
                edges_ary[con_ary[i, j], findfirst(x -> x == i, con_ary[con_ary[i, j],:])] = num_edges
                num_edges +=1
            end
        end
    end
    #print(typeof(edges_ary))
    return edges_ary
end

function get_edges_ary(nodes_ary::Array{Int64, 3})
    con_ary = nodes_ary[:, :, 1]
    return get_edges_ary(con_ary)
end

function get_node_dims_arys(con_ary::Array{Int64, 2}, e_weight::Array{Int64, 2})
    num_nodes = size(con_ary,1)


    arys = Vector{Int}[]
    for (i, nodes) in enumerate(eachrow(con_ary))
        ary = zeros(Int, num_nodes) 
        for (j, node) in enumerate(nodes)
            positive_nodes = nodes[nodes .>= 0]
            if con_ary[i, j] >0
                ary[node] =  e_weight[i, j] 
            end
        end
        push!(arys, ary)
        
    end
    
    return stack(arys)
end



function get_node_bool_arys(con_ary::Array{Int64, 2})
    num_nodes = size(con_ary,1)

    arys = Vector{Bool}[]
    for (i, nodes) in enumerate(eachrow(con_ary))
        ary = zeros(Bool, num_nodes) 
        ary[i] = true
        
        push!(arys, ary)
    end
    
    return stack(arys)
end

function matrixToAdjList(matrix)
    # Initialize an empty list of vectors to store the adjacency list
    adjList = Vector{Int}[]

    # Iterate through each row of the matrix
    for row in eachrow(matrix)
        # Filter out '-1' and add the remaining elements as a vector to the adjList
        push!(adjList, [element for element in row if element != -1])
    end

    return adjList


end
#Edge-weights are ordered when first looping over the list of lists 
#and then over this list only counted once with taking the upper triangle, 
#with the index of a column equal to the index of a list in the list of lists
function esizeFromNodesAry(con_ary, e_weigth)
    
    esize = Int[]
    
    for i in axes(con_ary, 1)
        for j in axes(con_ary, 2)
            if i>con_ary[i, j] && con_ary[i, j] !== -1
                push!(esize, e_weigth[i, j])
            end
            
        end
    end
    return esize'



end
function adjListToSparseAdjMatrix(adjacency_list)
    rows = Int[]
    cols = Int[]
    vals = Bool[]

    # Fill the row indices, column indices, and values
    for (i, neighbors) in enumerate(adjacency_list)
        for neighbor in neighbors
            push!(rows, neighbor)
            push!(cols, i)
            push!(vals, true)  # Using 1 for unweighted graphs
        end
    end

    return sparse(rows, cols, vals)
    
end

function get_nodes_from_edge(sg, edge)
    (e, r, _) = collect(edges(sg))
    i, j = findall(x -> x == edge, e)
    return r[i], r[j]
end

function get_edge_from_nodes(sg, i, j)
    (e, r, c) = collect(edges(sg))
    j = findfirst(x -> x== (i,j), [(r[i], c[i]) for i in 1:length(r)])
    return e[j]
end



## Example usage
#matrix = [
#    2 -1 -1 -1;
#    1 3 -1 -1;
#    2 4 -1 -1;
#    3 -1 -1 -1
#]
#
#adjList = matrixToAdjList(matrix)
#
## This will print the adjacency list
#println(adjList)
#function update(edge::Int, edges_ary, node_dims_arys::Matrix{Int64}, node_bool_arys::Matrix{Bool})
#    node_i0, node_i1 = first.(Tuple.(findall( x -> x == edge, edges_ary)))
#    print(node_i0)
#    
#
#    n0_B = node_bool_arys[node_i0,:]
#    n1_B = node_bool_arys[node_i1,:]
#
#    #n0 = node_dims_arys[node_i0,:]
#    #n1 = node_dims_arys[node_i1,:]
#    n0 = reduce(+, [node_dims_arys[i,:] for i in findall(n0_B)])
#    n1 = reduce(+, [node_dims_arys[i,:] for i in findall(n1_B)])
#    print(n0)
#    print(n1_B)
#    if_diff = !(n0_B[node_i1])
#    if if_diff
#        
#        contract_dims = n0 .+ n1
#        contract_dims[n0.*n1 .!= 0 ] .=n0[n0.*n1 .!= 0 ] .* n1[n0.*n1 .!= 0 ]
#        
#        contract_bool = n0_B .| n1_B
#        
#        flop = 1*prod(filter(x -> x != 0, n0[.!contract_bool]))*prod(filter(x -> x != 0, n1[.!contract_bool]))*prod(filter(x -> x != 0, n0[contract_bool]))
#        print(prod(n0[n1_B]))
#        flop = 1*prod(filter(x -> x != 0, n0[.!contract_bool]), init=1)*prod(filter(x -> x != 0, n1[.!contract_bool]), init=1)*prod(filter(x -> x != 0, n0[n1_B]), init=1)
#        
# 
#        contract_dims[contract_bool] .= 0 
#
#        
#        foreach(i -> node_dims_arys[i, :] .= fill!(similar(contract_dims), 0), findall(contract_bool))
#        foreach(i -> node_dims_arys[:, i] .= fill!(similar(contract_dims), 0), findall(contract_bool))
#        
#        node_dims_arys[:, node_i1] .= contract_dims
#        node_dims_arys[node_i1, :] .= contract_dims
#
#        #foreach(i -> node_dims_arys[i, :] .= contract_dims, findall(contract_bool))
#        #foreach(j -> node_dims_arys[:, j] .= contract_dims, findall(contract_bool))
#
#        foreach(i -> node_bool_arys[i, :] .= contract_bool, findall(contract_bool))
#        foreach(j -> node_bool_arys[:, j] .= contract_bool, findall(contract_bool))
# 
#
#        global flops += flop
#    end
#
#    
#
#end
#
#flops = 0
#con_ary = [2 5 -1; 1 3 5; 2 4 -1; 3 5 -1; 1 2 4]
#e_weight = [4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4]
#edges_ary = get_edges_ary(con_ary)
#print(typeof(edges_ary))
#print(edges_ary)
#print("\n ----------- \n")
#node_bool_arys = get_node_bool_arys(con_ary)
#
#node_dims_arys = get_node_dims_arys(con_ary, e_weight)
#
#
#
#
#board =  convert(Array{Float32}, cat(node_bool_arys, node_dims_arys, dims=3))
#
#
#
#print("\n ----------- \n")
#print("\n __________________________ \n")
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#update(1, edges_ary, node_dims_arys, node_bool_arys )
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#
#update(6, edges_ary, node_dims_arys, node_bool_arys )
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#
#update(3, edges_ary, node_dims_arys, node_bool_arys )
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#
#update(4, edges_ary, node_dims_arys, node_bool_arys )
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#
#print("\n end")
