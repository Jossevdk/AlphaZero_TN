#using Distributed
#addprocs(2)
#r = remotecall(rand, 2, 2, 2)
#fetch(r)
#print(r)
#print(Threads.threadpoolsize())
#function busywait(seconds)
#    tstart = time_ns()
#    while (time_ns() - tstart) / 1e9 < seconds
#    end
#end
#
#@time begin
#            Threads.@spawn busywait(5)
#            Threads.@threads :static for i in 1:Threads.threadpoolsize()
#                busywait(1)
#            end
#        end
#
#
#@time begin
#            Threads.@spawn busywait(5)
#            Threads.@threads :dynamic for i in 1:Threads.threadpoolsize()
#                busywait(1)
#            end
#        end
#______________________________________________
#
#function task(i)
#    print(Threads.threadid())
#    return string("do task ", i)
#end
#
#t = Threads.@spawn task(5)
#
#fetch(t)

####workers = [2, 3, 6]
####
####tasks = map(workers) do w
####    a = 2*w
####end
####
####print(tasks)
####
####
####
####
####struct mc 
####    v::Int
####end
####
####
####struct fun{mc_make, Int32}
####    a::mc_make
####    b::Int32
####end
####
####function lol(mc::mc)
####    return mc.v*mc.v
####end
####
####
####g = fun(3) do u
####    return mc(u)
####end
####
####print(g.a(3))
####print(lol(g.a(3)))
####
####
####actions = collect(1:6)
####mask = trues(6)
####mask[3] = false
####print(actions[mask])

A = [1, 2, 3, 7, 5]
B = copyto!(similar(A, 1, 5), collect(1:5))
tuple.(A, B)
