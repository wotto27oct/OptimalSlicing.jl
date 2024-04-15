using OptimalSlicing
using Test
using Polynomials

@testset "OptimalSlicing.jl" begin
    # Write your tests here.
    pathinfo, path_print = search("TEBD")
    @test pathinfo.cost == Polynomial([0, 0, 16, 10])
    @test pathinfo.max_size == Polynomial([0, 0, 4])
    pathinfo, path_print = search("3_1_1DTTN")
    @test pathinfo.cost == Polynomial([0, 0, 0, 0, 0, 0, 4])
    @test pathinfo.max_size == Polynomial([0, 0, 0, 0, 1])
end

nothing