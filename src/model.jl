# Geometry and CUFSM model construction: centerlines with fillets, the
# prop/node/elem arrays in CUFSM's standard layout, and the zero-thickness
# net-section (hole) model.

_interp_x(X, Y, i, y) = Y[i+1] == Y[i] ? X[i] : X[i] + (X[i+1] - X[i]) * (y - Y[i]) / (Y[i+1] - Y[i])

"""
    polyline_with_fillets(corners, r; n_arc = 4, n_flat = 1) -> (X, Y)

Centerline of an open thin-walled section from its corner points (a vector
of `(x, y)` tuples in order, first and last being the free edges), every
interior corner replaced by a circular fillet of centerline radius `r`
discretised into `n_arc` straight segments, and every flat subdivided into
`n_flat` elements. `r = 0` gives the sharp-cornered polyline. Fillets must
fit: the tangent length `r·tan(θ/2)` may not exceed half of either
adjoining flat.
"""
function polyline_with_fillets(corners, r; n_arc = 4, n_flat = 1)
    nc = length(corners)
    nc >= 2 || error("need at least two corner points")
    pts = [Float64.(collect(c)) for c in corners]
    # per interior corner: arc start, arc points, arc end (tangent points)
    starts = Vector{Vector{Float64}}(undef, nc); ends = Vector{Vector{Float64}}(undef, nc)
    arcs   = Dict{Int, Vector{Vector{Float64}}}()
    for k in 2:(nc - 1)
        u1 = pts[k] - pts[k-1]; u1 ./= hypot(u1...)
        u2 = pts[k+1] - pts[k]; u2 ./= hypot(u2...)
        cross = u1[1] * u2[2] - u1[2] * u2[1]
        θ = atan(cross, u1[1] * u2[1] + u1[2] * u2[2])
        if r == 0 || abs(θ) < 1e-9
            starts[k] = pts[k]; ends[k] = pts[k]; arcs[k] = Vector{Float64}[]
            continue
        end
        d = r * tan(abs(θ) / 2)
        (d <= hypot((pts[k] - pts[k-1])...) / 2 + 1e-12 && d <= hypot((pts[k+1] - pts[k])...) / 2 + 1e-12) ||
            error("fillet radius $r does not fit at corner $k")
        s = sign(θ)
        n1 = s * [-u1[2], u1[1]]                      # inward normal at the arc start
        a0 = pts[k] - d * u1
        center = a0 + r * n1
        φ0 = atan(a0[2] - center[2], a0[1] - center[1])
        starts[k] = a0; ends[k] = pts[k] + d * u2
        arcs[k] = [center + r * [cos(φ0 + θ * j / n_arc), sin(φ0 + θ * j / n_arc)] for j in 1:(n_arc - 1)]   # signed sweep
    end
    starts[1] = pts[1]; ends[1] = pts[1]; starts[nc] = pts[nc]; ends[nc] = pts[nc]

    X = Float64[pts[1][1]]; Y = Float64[pts[1][2]]
    for k in 1:(nc - 1)
        a, b = ends[k], starts[k+1]               # flat from the end of corner k to the start of corner k+1
        for j in 1:n_flat
            f = j / n_flat
            push!(X, a[1] + f * (b[1] - a[1])); push!(Y, a[2] + f * (b[2] - a[2]))
        end
        if k + 1 < nc && starts[k+1] != ends[k+1]    # a real fillet (sharp corners add no nodes)
            for p in arcs[k+1]
                push!(X, p[1]); push!(Y, p[2])
            end
            push!(X, ends[k+1][1]); push!(Y, ends[k+1][2])
        end
    end
    return X, Y
end

"""
    cufsm_model(X, Y, t, E; ν = 0.3, P = 1.0, Mxx = 0.0, Mzz = 0.0, M11 = 0.0, M22 = 0.0, mat = 100)
        -> (prop, node, elem)

Standard CUFSM arrays for an open, sequentially connected centerline
(`X[i], Y[i]` are the node coordinates in order; element i joins nodes i and
i+1). `t` is one thickness or a per-element vector. Every DOF is free and
the reference stress distribution comes from `CUFSM.stresgen` for the given
axial force and moments (unit values give load factors in force / moment
units). Consistent units throughout: coordinates and `t` in one length unit,
`E` in force/length², so the load factors come out in force (P) or
force·length (M).
"""
function cufsm_model(X, Y, t, E; ν = 0.3, P = 1.0, Mxx = 0.0, Mzz = 0.0, M11 = 0.0, M22 = 0.0, mat = 100)
    n = length(X)
    length(Y) == n || error("X and Y must have the same length")
    ne = n - 1
    tv = t isa AbstractVector ? collect(Float64, t) : fill(Float64(t), ne)
    length(tv) == ne || error("t must be a scalar or have one entry per element ($(ne))")
    coord = hcat(Float64.(X), Float64.(Y))

    ends_mat = zeros(ne, 3)
    for k in 1:ne
        ends_mat[k, 1] = k; ends_mat[k, 2] = k + 1; ends_mat[k, 3] = tv[k]
    end
    sp = CUFSM.cutwp_prop2(coord, ends_mat)

    node = zeros(Float64, n, 8)
    node[:, 1]   .= 1:n
    node[:, 2:3] .= coord
    node[:, 4:7] .= 1.0

    elem = zeros(Float64, ne, 5)
    elem[:, 1]   .= 1:ne
    elem[:, 2:4] .= ends_mat
    elem[:, 5]   .= mat

    G    = E / (2 * (1 + ν))
    prop = [Float64(mat) E E ν ν G]

    node = CUFSM.stresgen(node, P, Mxx, Mzz, M11, M22,
                          sp.A, sp.xc, sp.yc, sp.Ixx, sp.Iyy, sp.Ixy, sp.θ, sp.I1, sp.I2, 0)
    return prop, node, elem
end

"""
    straight_runs(X, Y; tol_deg = 1.0) -> Vector{UnitRange{Int}}

Element index ranges of the maximal straight (collinear within `tol_deg`)
runs of a centerline, in order.
"""
function straight_runs(X, Y; tol_deg = 1.0)
    ne = length(X) - 1
    runs = UnitRange{Int}[]
    s = 1
    for i in 2:ne
        d1x, d1y = X[i] - X[i-1], Y[i] - Y[i-1]
        d2x, d2y = X[i+1] - X[i], Y[i+1] - Y[i]
        if abs(atan(d1x * d2y - d1y * d2x, d1x * d2x + d1y * d2y)) > deg2rad(tol_deg)
            push!(runs, s:(i - 1)); s = i
        end
    end
    push!(runs, s:ne)
    return runs
end

"""
    zero_thickness_hole(X, Y, t, y_lo, y_hi; tol = 0.1 * t, web = nothing) -> (X, Y, t_vec)

Zero-thickness net-section model of a web hole spanning `y_lo ≤ y ≤ y_hi`
(Moen & Schafer's approach, as used for Pcrl_hole / Mcrl_hole): the
centerline nodes inside the span are removed and replaced by two boundary
nodes joined by ONE straight strip of zero thickness, so the two remaining
parts of the section are connected geometrically but not mechanically. Any
original node left within `tol` of a boundary node is dropped (boundary
nodes are kept) so no near-degenerate strip is created.

The hole is placed on the web: by default the straight run of elements with
the largest extent in y (see `straight_runs`); pass `web` as a range of
element indices to choose it explicitly (e.g. when a web stiffener splits
the flat). Lips and flanges that happen to cross the same y levels are
ignored.
"""
function zero_thickness_hole(X, Y, t, y_lo, y_hi; tol = 0.1 * t, web = nothing)
    y_lo < y_hi || error("y_lo must be below y_hi")
    n = length(X)
    if web === nothing
        runs = straight_runs(X, Y)
        web = runs[argmax([abs(Y[r.stop + 1] - Y[r.start]) for r in runs])]
    end
    ylo_web, yhi_web = extrema((Y[first(web)], Y[last(web) + 1]))
    (ylo_web <= y_lo && y_hi <= yhi_web) ||
        error("hole span $y_lo … $y_hi is not inside the web (y from $ylo_web to $yhi_web); pass `web` explicitly if the wrong flat was detected")
    crosses(i, y) = Y[i] <= y <= Y[i+1] || Y[i] >= y >= Y[i+1]
    i_lo = findfirst(i -> crosses(i, y_lo), web)
    i_hi = findfirst(i -> crosses(i, y_hi), web)
    (i_lo === nothing || i_hi === nothing) && error("hole bounds not found on the web")
    i_lo, i_hi = web[i_lo], web[i_hi]
    i_lo <= i_hi || ((i_lo, i_hi) = (i_hi, i_lo); (y_lo, y_hi) = (y_hi, y_lo))   # web running downward
    x_lo = _interp_x(X, Y, i_lo, y_lo)
    x_hi = _interp_x(X, Y, i_hi, y_hi)

    Xh = vcat(Float64.(X[1:i_lo]), x_lo, x_hi, Float64.(X[(i_hi + 1):end]))
    Yh = vcat(Float64.(Y[1:i_lo]), y_lo, y_hi, Float64.(Y[(i_hi + 1):end]))
    idx_lo = i_lo + 1              # hole element joins nodes idx_lo and idx_lo + 1
    while true
        seg = hypot.(diff(Xh), diff(Yh))
        k = findfirst(i -> seg[i] < tol && i != idx_lo, eachindex(seg))
        k === nothing && break
        j = k + 1 == idx_lo ? k : (k == idx_lo + 1 ? k + 1 : (k + 1 == length(Xh) ? k : k + 1))
        deleteat!(Xh, j); deleteat!(Yh, j)
        j < idx_lo && (idx_lo -= 1)
    end
    t_vec = fill(Float64(t), length(Xh) - 1)
    t_vec[idx_lo] = 0.0
    return Xh, Yh, t_vec
end
