using LinearAlgebra, Random

@testset "MPC Loop Test" begin
    N       = 10
    Nmpc    = 5
    Ns      = 2

    A = [0.921  0.0    0.041  0.0;
         0.0    0.918  0.0    0.033;
         0.0    0.0    0.924  0.0;
         0.0    0.0    0.0    0.937]

    B = [0.017  0.001;
         0.001  0.023;
         0.0    0.061;
         0.072  0.0]

    E = Matrix(I, 4, 4)

    nx, nu = size(B)
    nw = size(E, 2)

    Q    = Matrix(3.0*I, nx, nx)
    R    = Matrix(10^-4*I, nu, nu)
    QN   = Q

    Wi_OrthonoPoly = HermiteOrthonoPoly(2)
    Wi_coeff       = 0.05 .* [1, 1, sqrt(2)]
    WiPCE          = OrthonoPCE(Wi_OrthonoPoly, Wi_coeff)
    WPCE           = genPCE([WiPCE for _ = 1:nw])

    x0 = ones(nx)

    lbx = ([-2,-2,-Inf,-Inf], [0.1,0.1,1,1])
    ubx = ([2,2,Inf,Inf], [0.1,0.1,1,1])

    problem = defineOCP(
        N, 
        A, B, E,
        x0, WPCE.coeff;
        Q=Q, R=R, QN=QN,
        lbx=lbx, ubx=ubx
    )

    model = buildOCP(problem; print_level=0, max_cpu_time=10.)

    Random.seed!(1)

    xTrajs = Array{Float64}(undef, nx, Nmpc+1, Ns)
    xTrajs[:,1,:] = 2 .* rand(nx, Ns) .- 1

    uTrajs = Array{Float64}(undef, nu, Nmpc, Ns)

    ξTrajs = randn(nw, Nmpc, Ns)
    wTrajs = 0.05 .* (ξTrajs .+ ξTrajs.^2)

    @testset "MPC Execution" begin
        ok = true
        try
            con_initial_param(model)
            for s = 1:Ns
                for k = 1:Nmpc
                    update_initial_param(model, xTrajs[:,k,s])
                    _, usol, _ = solveOCP(model)
                    uTrajs[:,k,s] = usol[:,1,1]
                    xTrajs[:,k+1,s] = A*xTrajs[:,k,s] + B*uTrajs[:,k,s] + E*wTrajs[:,k,s]
                end
            end
        catch e
            @warn "MPC test failed with error: $e"
            ok = false
        end
        @test ok == true
    end

    @test size(xTrajs) == (nx, Nmpc+1, Ns)
    @test size(uTrajs) == (nu, Nmpc, Ns)

    @test all(isfinite, xTrajs)
    @test all(isfinite, uTrajs)
end
