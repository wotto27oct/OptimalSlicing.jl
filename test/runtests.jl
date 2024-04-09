using OptimalSlicing
using Test
using Polynomials

@testset "OptimalSlicing.jl" begin
    # Write your tests here.
    cost, path, max_size, slices = search("TEBD")
    @test cost == Polynomial([0, 0, 16, 10])
    @test max_size == Polynomial([0, 0, 4])
    cost, path, max_size, slices = search("3_1_1DTTN")
    @test cost == Polynomial([0, 0, 0, 0, 0, 0, 4])
    @test max_size == Polynomial([0, 0, 0, 0, 1])
end

nothing