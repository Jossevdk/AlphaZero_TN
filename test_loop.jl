    
using Statistics
using Revise
using AlphaZero
using EinExprs
using GLMakie, GraphMakie
using JSON
using SparseArrayKit
using SparseArrays



function read_json_file(filename)
    open(filename, "r") do f
        return JSON.parse(f)
    end
end

function parse_tuple(s::String)
  s = replace(s, "(" => "", ")" => "", " " => "")
  return Tuple(parse(Int, x) for x in split(s, ","))
end


data = read_json_file("25_tensors_3mean.json")

costs = []
for i in 1:1
    str_dict = data[i]["eq_$(i-1)"]


    tuple_dict = Dict(parse_tuple(k) => v for (k, v) in str_dict)
    I = [i[1]+1 for (i, v) in tuple_dict]
    J = [i[2]+1 for (i, v) in tuple_dict]
    V = [v for (i, v) in tuple_dict]
    A = sparse(I, J, V)
    sizes = reshape(data[i]["eq_size_$(i-1)"], 1, :)  

    tensors = EinExpr{Symbol}[]
    list = [Symbol(Char(i)) for i in 97:200]
    list_of_tensors = [ Symbol[] for i in 1:size(A, 1)]
    k=1
    l=1
    sizedict = Dict{Symbol, Float32}()
    for i in 1:size(A, 1)
        for j in i:size(A, 2)
            if A[i, j] == 1
                push!(list_of_tensors[i], list[k])
                push!(list_of_tensors[j], list[k])
                if j>i
                    sizedict[list[k]] = sizes[1, l]
                    l+=1
                end  
                k += 1
            end
        end
    end
    for i in 1:size(A, 1)
        push!(tensors, EinExpr(list_of_tensors[i]))
    end

    expr = sum(tensors)
    path = SizedEinExpr(expr, sizedict)

    cost = -einexpr(Exhaustive(metric=flops, strategy=:depth), path)[2]
    push!(costs, cost)
end

print(costs, mean(costs))
