"""
This module provides utilities to build neural networks with Flux,
along with a library of standard architectures.
"""
module FluxLib

export SimpleNet, SimpleNetHP, ResNet, ResNetHP

using ..AlphaZero
using GeometricFlux
using CUDA
using Base: @kwdef

import Flux

CUDA.allowscalar(false)
array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

using Flux: relu, softmax, flatten
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection
import Zygote

#####
##### Flux Networks
#####

"""
    FluxNetwork <: AbstractNetwork

Abstract type for neural networks implemented using the _Flux_ framework.

The `regularized_params_` function must be overrided for all layers containing
parameters that are subject to regularization.

Provided that the above holds, `FluxNetwork` implements the full
network interface with the following exceptions:
[`Network.HyperParams`](@ref), [`Network.hyperparams`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref).
"""
abstract type FluxNetwork <: AbstractNetwork end

function Base.copy(nn::Net) where Net <: FluxNetwork
  #new = Net(Network.hyperparams(nn))
  #Flux.loadparams!(new, Flux.params(nn))
  #return new
  return Base.deepcopy(nn)
end

Network.to_cpu(nn::FluxNetwork) = Flux.cpu(nn)

function Network.to_gpu(nn::FluxNetwork)
  CUDA.allowscalar(false)
  return Flux.gpu(nn)
end

function Network.set_test_mode!(nn::FluxNetwork, mode)
  Flux.testmode!(nn, mode)
end


Network.convert_input(nn::FluxNetwork, x) =
  Network.on_gpu(nn) ? Flux.gpu(x) : x

Network.convert_output(nn::FluxNetwork, x) = Flux.cpu(x)

Network.params(nn::FluxNetwork) = Flux.params(nn)

# This should be included in Flux
function lossgrads(f, args...)
  
  val, back = Zygote.pullback(f, args...)
  grad = back(Zygote.sensitivity(val))
  return val, grad
end

function Network.train!(callback, nn::FluxNetwork, opt::Adam, loss, data, n)
  optimiser = Flux.Adam(opt.lr)
  params = Flux.params(nn)
  gspec = nn.gspec
  
  for (i, d) in enumerate(data)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    callback(i, l)
  end
end

function Network.train!(
    callback, nn::FluxNetwork, opt::CyclicNesterov, loss, data, n)
  lr = CyclicSchedule(
    opt.lr_base,
    opt.lr_high,
    opt.lr_low, n=n)
  momentum = CyclicSchedule(
    opt.momentum_high,
    opt.momentum_low,
    opt.momentum_high, n=n)
  optimiser = Flux.Nesterov(opt.lr_low, opt.momentum_high)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    optimiser.eta = lr[i]
    optimiser.rho = momentum[i]
    callback(i, l)
  end
end

regularized_params_(l) = []
regularized_params_(l::Flux.Dense) = [l.weight]
regularized_params_(l::Flux.Conv) = [l.weight]

function Network.regularized_params(net::FluxNetwork)
  return (w for l in Flux.modules(net) for w in regularized_params_(l))
end

function Network.gc(::FluxNetwork)
  GC.gc(true)
  # CUDA.reclaim()
end

#####
##### Common functions between two-head neural networks
#####

"""
    TwoHeadNetwork <: FluxNetwork

An abstract type for two-head neural networks implemented with Flux.

Subtypes are assumed to have fields
`hyper`, `gspec`, `common`, `vhead` and `phead`. Based on those, an implementation
is provided for [`Network.hyperparams`](@ref), [`Network.game_spec`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref), leaving only
[`Network.HyperParams`](@ref) to be implemented.
"""
abstract type TwoHeadNetwork <: FluxNetwork end


function count_elements(arr)
  counts = Dict{eltype(arr), Int64}()
  for element in arr
      if haskey(counts, element)
          counts[element] += 1
      else
          counts[element] = 1
      end
  end
  return counts
end

function Network.forward(nn::TwoHeadNetwork, state::BatchFeatureGraph, actions_mask = nothing)
  # print(array_on_gpu(nn.vhead[end].weight))
  # print(typeof((nn)))
  # print(array_on_gpu(node_feature(state)))
  c = nn.common(state)

  # print(array_on_gpu(node_feature(c)))
  c = c.fg
  p_ = nn.phead(c)
  v = nn.vhead(c)
  
 
  if array_on_gpu(p_)
    p_ = permutedims(p_, (2, 1))  # Transpose p
  else
    p_ = p_' |> collect
  end


  graph_indicator = state.graph_indicator
  # Assuming graph_indicator is sorted in ascending order
  cumcounts = cumsum([count(==(x), graph_indicator) for x in sort(unique(graph_indicator))])
  parts = [vcat(softmax(view(p_, i:j)), zeros(size(actions_mask, 1) - length(view(p_, i:j)))) for (i, j) in zip([1; cumcounts[1:end-1] .+ 1], cumcounts)]
  p = hcat(parts...)
  if array_on_gpu(v)
    
    v = map(((i, j),) -> sum(view(v, i:j)), zip([1; cumcounts[1:end-1] .+ 1], cumcounts)) |> CUDA.CuArray
    v = reshape(v,1,:)
  else
    #v = [tanh(sum(view(v, i:j))) for (i, j) in zip([1; cumcounts[1:end-1] .+ 1], cumcounts)]' |> collect
    v = [sum(view(v, i:j)) for (i, j) in zip([1; cumcounts[1:end-1] .+ 1], cumcounts)]' |> collect

  end
  #print("\n graph_indicator used \n")

  
  #print(("\n After NN: ", p, v, "\n ======================= \n"))
  return (p, v)
end


function Network.forward(nn::TwoHeadNetwork, state, graph_indicator = nothing, actions_mask = nothing)
  # print(array_on_gpu(nn.vhead[end].weight))
  # print(typeof((nn)))
  # print(array_on_gpu(node_feature(state)))
  c = nn.common(state)
  # print(array_on_gpu(node_feature(c)))
  p_ = nn.phead(c)
  p_ = softmax(p_, dims=2)
  v = nn.vhead(c)
  

  if array_on_gpu(p_)
    v = CUDA.fill(sum(v), 1, 1)
    p = permutedims(p_, (2, 1))  # Transpose p
  else
    p = p_' |> collect
    #v = [tanh(sum(v))]' |> collect
    v = [sum(v)]' |> collect
  end

   
  
  #print(("\n After NN: ", p, v, "\n ======================= \n"))
  return (p, v)
end

# Flux.@functor does not work with abstract types
function Flux.functor(nn::Net) where Net <: TwoHeadNetwork
  children = (nn.common, nn.vhead, nn.phead)
  constructor = cs -> Net(nn.gspec, nn.hyper, cs...)
  return (children, constructor)
end

Network.hyperparams(nn::TwoHeadNetwork) = nn.hyper

Network.game_spec(nn::TwoHeadNetwork) = nn.gspec

Network.on_gpu(nn::TwoHeadNetwork) = array_on_gpu(nn.vhead[end].bias)

#####
##### Include networks library
#####

include("architectures/simplenet.jl")
include("architectures/resnet.jl")
include("architectures/graphnet.jl")

end
