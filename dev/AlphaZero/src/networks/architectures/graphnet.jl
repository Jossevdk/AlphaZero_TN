using GeometricFlux

import GeometricFlux: MessagePassing, GraphSignals
using Flux
using Flux: glorot_uniform, @functor

using SparseArrays
using NNlib
using Statistics
# Create a new convolution layer for edges

_gather(::Nothing, idx) = nothing
# _gather(A::Fill{T,2,Axes}, idx) where {T,Axes} = fill(A.value, A.axes[1], length(idx))
_gather(A::AbstractMatrix, idx) = NNlib.gather(A, idx)
_gather(A::AbstractArray, idx) = NNlib.gather(A, batched_index(idx, size(A)[end]))

_scatter(aggr, E, xs::AbstractArray) = NNlib.scatter(aggr, E, xs)
_scatter(aggr, E, xs::AbstractArray, dstsize) = NNlib.scatter(aggr, E, xs; dstsize=dstsize)

_matmul(A::AbstractMatrix, B::AbstractMatrix) = A * B
_matmul(A::AbstractArray, B::AbstractArray) = NNlib.batched_mul(A, B)



struct GraphEConv{A<:AbstractMatrix,B,F,O} <: MessagePassing
    weight1::A
    weight2::A
    bias::B
    σ::F
    naggr::O
    eaggr
end


function GraphEConv(nef::Int, nnf::Int, out::Int, σ=identity, naggr=mean, eaggr=+;
    init=glorot_uniform, bias::Bool=true)

W1 = init(out, nnf)
W2 = init(out, 2*nnf + nef)
b = Flux.create_bias(W1, bias, out)
GraphEConv(W1, W2, b, σ, naggr, eaggr)
end

@functor GraphEConv

Flux.trainable(l::GraphEConv) = (l.weight1, l.weight2, l.bias)

message(gc::GraphEConv, x_i, x_j::AbstractArray, e_ij) = _matmul(gc.weight2, vcat(e_ij, x_i, x_j))

update(gc::GraphEConv, m::AbstractArray, x::AbstractArray) = gc.σ.(_matmul(gc.weight1, x) .+ m .+ gc.bias)

# For variable graph
function (l::GraphEConv)(fg::AbstractFeaturedGraph)
nf = node_feature(fg)
ef = edge_feature(fg)
GraphSignals.check_num_nodes(fg, nf)
GraphSignals.check_num_edges(fg, ef)
E, V, _ = propagate(l, graph(fg), ef, nf, nothing, l.naggr, l.eaggr, nothing)
return ConcreteFeaturedGraph(fg, nf=V, ef=E)
end

function update_edge(gn::GraphEConv, e, vi, vj, u)
    return message(gn, vi, vj, e)
end

function update_batch_edge(gn::GraphEConv, el::NamedTuple, E, V, u)
    return update_edge(
        gn,
        _gather(E, el.es),
        _gather(V, el.xs),
        _gather(V, el.nbrs),
        u
    )
end

function aggregate_neighbors(gn::GraphEConv, el::NamedTuple, naggr, E)
    return _scatter(naggr, E, el.nbrs)
end

function update_batch_vertex(gn::GraphEConv, ::NamedTuple, Ē, V, u)
    return update(gn, Ē, V)
end

function propagate(gn::GraphEConv, sg::SparseGraph, E, V, u, naggr, eaggr, vaggr)
    el = GraphSignals.to_namedtuple(sg)
    return propagate(gn, el, E, V, u, naggr, eaggr, vaggr)
end

function propagate(gn::GraphEConv, el::NamedTuple, E, V, u, naggr, eaggr, vaggr)
    E = update_batch_edge(gn, el, E, V, u)
    Ē = aggregate_neighbors(gn, el, naggr, E)
    V = update_batch_vertex(gn, el, Ē, V, u)

    E = 0.5f0*_scatter(eaggr, E, el.es)
    
    return E, V, u
end


@kwdef struct GraphNetHP
    
    hidden_dim::Int
    output_dim::Int
    use_batch_norm :: Bool = false
    batch_norm_momentum :: Float32 = 0.6f0
end

# Define the GraphNet architecture
mutable struct GraphNet <: TwoHeadNetwork
    gspec
    hyper
    common
    vhead
    phead
end

function sum_to_vector_layer(x)
    return [mapreduce(identity, +, x)]
end

function GraphNet(gspec::AbstractGameSpec, hyper::GraphNetHP)

    input_dim = GI.features_dim(gspec)
    common = Chain(
        GraphEConv(2, 1, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 5),
        GraphEConv(5, 5, 2)
    )
    
    phead = Chain(
        edge_feature, Dense(hyper.hidden_dim, 1, tanh), softmax
    )
    vhead = Chain(edge_feature, 
        Dense(hyper.hidden_dim, 1, tanh),
        
    )
    
    return GraphNet(gspec, hyper, common, vhead, phead)
end


Network.HyperParams(::Type{GraphNet}) = GraphNetHP

function Base.copy(nn::GraphNet)
    return GraphNet(
        nn.gspec,
        nn.hyper,
        deepcopy(nn.common),
        deepcopy(nn.vhead),
        deepcopy(nn.phead)
    )
end
