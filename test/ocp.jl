using LinearAlgebra

@testset "Basic Stochastic OCP Test" begin
    # Problem setup
    N    = 50 
    A    = [0.95123 0;
            0.08833 0.81873]
    B    =[-0.0048771; -0.0020429]
    E    = [1.0; 1.0]

    Q    = Matrix(1.0*I, 2, 2)
    R    = 1.0
    QN   = Q

    ubx = ([Inf; 0.24], [1; 0.1])

    # Define X0 and W and generate their PCEs
    X0  = [GaussMeasureParametric(0.5,0.05); GaussMeasureParametric(0.1,0.01)]
    W   = [UniformMeasureParametric(-0.0173, 0.0173)]
    X0PCE   = genPCE(X0)
    WPCE    = genPCE(W)

    # define the stochastic OCP
    problem = defineOCP(
        N, 
        A, B, E,
        X0PCE.coeff, WPCE.coeff;
        Q=Q, R=R, QN=QN,
        ubx = ubx
    )

    # Basic checks on problem
    @test problem.N == N
    @test problem.nx == size(A,1)
    @test problem.nu == size(B,2)
    @test problem.nw == size(E,2)
    @test typeof(problem) == StochProb

    # build the JuMP model
    model = buildOCP(problem; print_level=0)

    # solve the stochastic OCP
    xsolPCE, usolPCE, obj = solveOCP(model)
    
    # Basic sanity checks on outputs
    @test size(xsolPCE,1) == problem.nx
    @test size(usolPCE,1) == problem.nu
    @test obj isa Real
end
