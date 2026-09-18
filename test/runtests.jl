using Test
using LinearAlgebra
using BucklingModeIdentification

# Lipped Cee, inches: 2.5 deep, 1.5 flanges, 0.5 lips, t = 0.05, E = 29500 ksi
const CEE = [(1.5, 0.5), (1.5, 0.0), (0.0, 0.0), (0.0, 2.5), (1.5, 2.5), (1.5, 2.0)]
const T   = 0.05
const E   = 29500.0

@testset "polyline_with_fillets" begin
    X, Y = polyline_with_fillets(CEE, 0.0; n_flat = 4)
    @test length(X) == 5 * 4 + 1
    Xr, Yr = polyline_with_fillets(CEE, 0.1; n_arc = 4, n_flat = 4)
    @test length(Xr) == length(X) + 4 * 4          # each fillet adds n_arc nodes
    seg = hypot.(diff(Xr), diff(Yr))
    @test all(seg .> 0.02)                          # no degenerate strips
    @test_throws ErrorException polyline_with_fillets(CEE, 5.0)
end

@testset "sharp lipped Cee reduces to classic cFSM" begin
    X, Y = polyline_with_fillets(CEE, 0.0; n_flat = 4)
    prop, node, elem = cufsm_model(X, Y, T, E)
    classes, primary = corner_topology(node, elem)
    @test length(classes) == size(node, 1) && count(primary) == 6   # 4 corners + 2 free edges
    r = decompose_modes(node, elem, prop, [1.0, 8.0, 100.0]; pure_subspaces = (:L, :D))
    @test r.counts.G == 4 && r.counts.D == 2 && r.counts.corner == 0
    @test size(r.participation) == (3, 4)
    @test all(abs.(vec(sum(r.participation; dims = 2)) .- 100) .< 1e-8)
    @test all(r.pure[:L] .>= r.load_factor .* (1 - 1e-8))     # constrained minima never below unconstrained
    @test all(r.pure[:D] .>= r.load_factor .* (1 - 1e-8))
    @test r.participation[3, 1] > 90                          # 100 in.: global
    lfL, shL = pure_mode_curve(node, elem, prop, [1.0, 8.0], [:L]; return_shapes = true)
    @test isapprox(lfL, r.pure[:L][1:2]; rtol = 1e-6)          # same eigenvalues as the in-decomposition pure curve
    @test length(shL[1]) == 4 * size(node, 1) && maximum(abs, shL[1]) ≈ 1
    rl = decompose_modes(node, elem, prop, [1.0]; corner_model = :literal)
    @test rl.counts == r.counts                                # no secondary nodes on a sharp mesh
    @test isapprox(rl.participation[1, :], r.participation[1, :]; atol = 1e-4)
end

@testset "rounded lipped Cee: elastic corners keep the sharp topology" begin
    n_arc = 4
    X, Y = polyline_with_fillets(CEE, 0.1; n_arc, n_flat = 4)
    prop, node, elem = cufsm_model(X, Y, T, E)
    r = decompose_modes(node, elem, prop, [1.0, 8.0, 100.0]; pure_subspaces = (:L,))
    @test r.counts.G == 4 && r.counts.D == 2                   # one primary warping per corner arc
    @test r.counts.corner == 4 * n_arc                         # the arcs' secondary warpings
    @test all(abs.(vec(sum(r.participation; dims = 2)) .- 100) .< 1e-8)
    @test all(r.pure[:L] .>= r.load_factor .* (1 - 1e-8))
    @test r.participation[3, 1] > 90
    rl = decompose_modes(node, elem, prop, [1.0]; corner_model = :literal)
    @test rl.counts.D == 4 * (n_arc + 1) + 2 - 4               # every fillet node a main node
end

@testset "geometry helpers" begin
    X, Y = polyline_with_fillets(CEE, 0.0; n_flat = 2)
    insert!(X, 3, X[2] + 1e-4); insert!(Y, 3, Y[2])          # a degenerate 1e-4 strip
    Xc, Yc = drop_short_segments(X, Y; min_length = 0.1 * T)
    @test length(Xc) == length(X) - 1
    Xh, Yh, tv = zero_thickness_hole(Xc, Yc, T, 0.75, 1.75)
    @test count(==(0.0), tv) == 1
    k = findfirst(==(0.0), tv)
    @test Yh[k] ≈ 0.75 && Yh[k+1] ≈ 1.75
    @test length(tv) == length(Xh) - 1
    # hole edges at y levels the lips also cross (lips span 0–0.5 and 2.0–2.5): must land on the web (x = 0)
    Xh, Yh, tv = zero_thickness_hole(Xc, Yc, T, 0.3, 2.2)
    k = findfirst(==(0.0), tv)
    @test count(==(0.0), tv) == 1 && Xh[k] == 0.0 && Xh[k+1] == 0.0 && Yh[k] ≈ 0.3 && Yh[k+1] ≈ 2.2
    @test_throws ErrorException zero_thickness_hole(Xc, Yc, T, -0.5, 1.0)     # outside the web
    @test length(straight_runs(Xc, Yc)) == 5
end

@testset "high-level API" begin
    X, Y = polyline_with_fillets(CEE, 0.1; n_arc = 4, n_flat = 4)
    Ls = [1.0, 8.0, 100.0]
    c1 = decompose_section(X, Y, T, E; name = "Cee", load = :axial, lengths = Ls)
    @test c1.model_key == "Pcr_gross" && c1.result.lengths == Ls
    c2 = decompose_section(X, Y, T, E; name = "Cee", load = :bending, hole = 1.0, lengths = Ls)
    @test c2.model_key == "Mcr_hole"
    @test count(==(0.0), c2.result.elem[:, 4]) == 1
    summarize(c1; io = devnull)
    tmp = mktempdir()
    paths = write_outputs(c1; outdir = tmp, figures = false)
    @test isfile(paths.json) && isfile(paths.viewer)
    html = read(paths.viewer, String)
    @test !occursin("__DATA_JSON__", html) && !occursin("__TITLE__", html)
    viewer = build_viewer([c1, c2]; outdir = tmp, title = "Test")
    @test isfile(viewer)
    @test isfile(joinpath(tmp, "data", "Cee__Pcr_gross.js")) && isfile(joinpath(tmp, "data", "Cee__Mcr_hole.js"))
    @test !plotting_available()                                # no Makie loaded in the test env
end
