"""
BucklingModeIdentification — modal identification of conventional (unconstrained)
finite strip buckling modes into Global / Distortional / Local / Other
participations, following Ádány & Schafer (2006, 2008) and the classification
built into CUFSM -- on the member's ACTUAL rounded-corner geometry.

The constrained finite strip method (cFSM) CONSTRAINS the eigenproblem to one
mechanical subspace and returns a "pure" mode curve. This package does the
opposite: it leaves the conventional signature curve exactly as it is and, at
every half-wavelength, expresses the actual buckled shape `d` in the basis of
cFSM's mechanical base vectors,

    d = [R_G  R_D  R_L  R_O] * c ,

so that the coefficient vector `c` says how much of the mode lives in each
subspace (participation) and `d_M = R_M * c_M` is the part of the buckled
shape belonging to subspace M -- the four component shapes sum back to `d`.

Rounded corners. cFSM's G and D criteria (Vlasov: no membrane shear, no
transverse membrane strain; warping present) give every non-collinear node
an independent warping value. On a rounded mesh that is every fillet node,
so the strictly strain-free ("literal") Vlasov space is large: for this
catalog's lipped Cees 48 warping DOFs instead of 16, i.e. 44 D modes instead
of 12, most of them "corner-warping" patterns that vary within a fillet.
Left as they are they wreck the identification: a flange-bowing local mode
is reproduced almost entirely by dozens of nearly parallel corner-warping
vectors and reads as ~97 % "distortional". The package therefore builds the
literal strain-free base and then, at every half-wavelength, treats ONE
warping value per corner arc (the node at the arc's angular midpoint) plus
the two free edges as PRIMARY and statically condenses the remaining
arc-node warpings through the section's true elastic stiffness projected on
the Vlasov space (`Kred = Tᵀ K T`): each rounded corner takes the
minimum-energy shape compatible with the flats -- the discrete equivalent of
Beregszászi & Ádány's rounded-corner cFSM with elastic corner elements. The
condensed space, made Ryd-orthogonal to the four G modes, is the physical D
space (same dimension as for the sharp-corner topology); the leftover
corner-warping directions are short-wavelength bending of the corner arcs
and are grouped with L, but their share is reported separately
(`corner_participation`). Every base vector stays exactly membrane-strain-
free, the geometry is never altered, and for a sharp-cornered mesh (no
secondary arc nodes) the construction reduces to classic cFSM.
`corner_model = :literal` skips the condensation for comparison.

Participation convention: the coefficients are solved exactly (the four
subspaces together span the full DOF space, so the base matrix is square) and
participation of class M is `‖d_M‖₂ / Σ_k ‖d_k‖₂`, the 2-norm of the class's
component displacement `d_M = R_M c_M` over the sum of the four. `d_M` is the
unique projection onto subspace M, so this measure does not depend on how
each subspace is spanned or how many vectors span it. CUFSM's classic
alternative -- the L1 norm of the coefficients themselves -- is not usable
with this base: the ~150 nearly parallel L/corner vectors versus 4 G vectors
inflate whichever class has more columns (a pure global mode read 69 %
"Local"). The column normalisation (`normalization = :vector` or
`:strain_energy`) only affects the coefficient values, not the participation.
"""
module BucklingModeIdentification

using LinearAlgebra
using CUFSM
using cFSM
using CUFSMModalGeometry
using JSON3

export decompose_modes, characteristic_minima, discretized_shape, corner_topology, drop_short_segments,
       pure_mode_curve, plot_decomposition, plot_mode_components,
       export_viewer_json, write_viewer_html, write_profile_js, write_multi_viewer_html
export cufsm_model, zero_thickness_hole, straight_runs, polyline_with_fillets, SectionCase, decompose_section,
       summarize, plotting_available, write_outputs, build_viewer

const MODE_CLASSES = (:G, :D, :L, :O)
const MODE_LABELS  = (G = "Global", D = "Distortional", L = "Local", O = "Other")
const MODE_COLORS  = (G = :gray40, D = :darkorange, L = :dodgerblue, O = :mediumpurple)

struct DecompositionResult
    lengths::Vector{Float64}
    load_factor::Vector{Float64}           # decomposed mode's load factor at each length
    participation::Matrix{Float64}         # nlengths × 4 (G, D, L, O), percent, rows sum to 100
    corner_participation::Vector{Float64}  # diagnostic only: percent (of the same total) on the corner-arc warping directions, already included in L. Not physically separable from L -- these directions are the only base vectors that move a fillet node in-plane, so they carry most of any flange-bowing mode's coefficients.
    coefficients::Vector{Vector{Float64}}  # per length: full coefficient vector c
    shapes::Vector{Vector{Float64}}        # per length: the decomposed conventional mode d (CUFSM DOF order)
    components::Vector{Matrix{Float64}}    # per length: ndof × 4, columns d_G d_D d_L d_O (sum to d)
    counts::NamedTuple{(:G, :D, :corner, :L, :O), NTuple{5, Int}}   # subspace dimensions (corner ⊂ L)
    pure::Dict{Symbol, Vector{Float64}}    # constrained (cFSM) curves requested via `pure_subspaces`, e.g. :L, :D, Symbol("G+D")
    classes::Vector{Int}                   # per node, literal cFSM class: 1 bend (main), 2 free edge, 3 collinear sub-node
    primary::Vector{Bool}                  # per node: carries an independent (primary) warping value
    corner_model::Symbol
    node::Matrix{Float64}
    elem::Matrix{Float64}
    prop::Matrix{Float64}
    normalization::Symbol
    eig::Int
end

# ── Corner topology on the real geometry ─────────────────────────────────────

"""
    corner_topology(node, elem; angle_tol_deg = 5.0) -> (classes, primary)

For an open, sequentially connected section (element k joins nodes k and
k+1): `classes` is the literal cFSM class of every node (1 = bend/main node,
2 = free edge, 3 = collinear sub-node) and `primary` marks the nodes that
carry an independent warping value in the elastic-corner model -- the two
free edges and, for every corner ARC, the single node at its angular
midpoint (where the arc's accumulated turn first reaches half its total).

A node whose signed turn angle exceeds `angle_tol_deg` is a bend node;
consecutive bend nodes turning the same way form one arc (a rounded fillet,
a rib flank, a rib crest, ...). Two consecutive arcs turning opposite ways
(a rib flank meeting its crest, an S-shaped jog) are separate arcs even with
no straight node between them.
"""
function corner_topology(node, elem; angle_tol_deg = 5.0)
    n = size(node, 1)
    for k in 1:size(elem, 1)
        (Int(elem[k, 2]) == k && Int(elem[k, 3]) == k + 1) || error(
            "corner_topology expects sequential open-chain connectivity (element k joins nodes k, k+1)")
    end
    x = node[:, 2]; z = node[:, 3]
    turn = zeros(n)
    for i in 2:(n - 1)
        ax, az = x[i] - x[i-1], z[i] - z[i-1]
        bx, bz = x[i+1] - x[i], z[i+1] - z[i]
        turn[i] = atan(ax * bz - az * bx, ax * bx + az * bz)
    end
    tol = deg2rad(angle_tol_deg)

    classes = fill(3, n); classes[1] = 2; classes[n] = 2
    primary = falses(n); primary[1] = true; primary[n] = true
    i = 2
    while i <= n - 1
        if abs(turn[i]) > tol
            s = sign(turn[i]); j = i
            while j + 1 <= n - 1 && abs(turn[j+1]) > tol && sign(turn[j+1]) == s
                j += 1
            end
            classes[i:j] .= 1
            total = sum(abs, turn[i:j]); cum = 0.0; k = j
            for k2 in i:j
                cum += abs(turn[k2])
                if cum >= total / 2 - 1e-12
                    k = k2; break
                end
            end
            primary[k] = true
            i = j + 1
        else
            i += 1
        end
    end
    return classes, primary
end

"""
    drop_short_segments(X, Y; min_length) -> (X, Y)

Removes centerline nodes that sit closer than `min_length` to their
predecessor, so no finite strip is shorter than that. This is mesh hygiene,
not shape idealisation: a strip a few percent of the thickness long (the
Scottsdale 46 mm-flange Cees carry one 0.0008 in. zig-zag element in the web
jog) has bending stiffness ~1/b³, i.e. ~10⁹ times its neighbours, which
ill-conditions cFSM's planar solve and the corner condensation -- the global
mode then reads as "Local" and pure-D comes out 20× too stiff -- and makes
the conventional eigen-solve itself noisy at long half-wavelengths. Dropping
the node moves the centerline by less than `min_length`. A sensible
`min_length` is ~0.1 t: real fillet facets here are ≥ 0.35 t.
"""
function drop_short_segments(X, Y; min_length)
    X = collect(Float64, X); Y = collect(Float64, Y)
    while true
        seg = hypot.(diff(X), diff(Y))
        k = findfirst(<(min_length), seg)
        k === nothing && break
        j = k + 1 == length(X) ? k : k + 1        # keep the free-edge tip node
        deleteat!(X, j); deleteat!(Y, j)
    end
    return X, Y
end

"""
cFSM's main-node / sub-node bookkeeping (`m_node`, `m_elem`, `node_prop`)
built from an EXPLICIT per-node class list instead of cFSM's own
collinearity-tolerance test -- otherwise a transcription of
`cFSM._meta_elems`, so everything downstream in cFSM (constraint matrices,
DOF permutation, base vectors) works unchanged.
"""
function meta_elems_from_classes(node, elem, classes)
    nnode = size(node, 1); nelem = size(elem, 1)
    length(classes) == nnode || error("classes has $(length(classes)) entries for $nnode nodes")

    node_prop = zeros(Int, nnode, 4)
    node_prop[:, 1] = 1:nnode
    for i in 1:nnode
        mel = count(j -> Int(elem[j, 2]) == i || Int(elem[j, 3]) == i, 1:nelem)
        node_prop[i, 4] = classes[i]
        if classes[i] == 3
            mel == 2 || error("node $i is classed as a sub-node but has $mel adjacent elements (expected 2)")
            node_prop[i, 3] = 0
        elseif classes[i] == 2
            mel == 1 || error("node $i is classed as a free edge but has $mel adjacent elements (expected 1)")
            node_prop[i, 3] = mel
        elseif classes[i] == 1
            mel >= 2 || error("node $i is classed as a corner but has $mel adjacent elements (expected >= 2)")
            node_prop[i, 3] = mel
        else
            error("node class must be 1 (corner), 2 (edge) or 3 (sub-node); node $i has $(classes[i])")
        end
    end

    MAX_SUB = 60
    m_el = zeros(Int, nelem, 4 + MAX_SUB)
    m_el[:, 1] = 1:nelem
    for j in 1:nelem
        m_el[j, 2] = Int(elem[j, 2]); m_el[j, 3] = Int(elem[j, 3])
    end
    for i in 1:nnode
        node_prop[i, 3] == 0 || continue
        els = [j for j in 1:nelem if m_el[j, 2] == i || m_el[j, 3] == i]
        length(els) == 2 || error("Sub-node $i has $(length(els)) adjacent elements (expected 2).")
        e1, e2 = els
        no1 = m_el[e1, 2]; no1 == i && (no1 = m_el[e1, 3])
        no2 = m_el[e2, 2]; no2 == i && (no2 = m_el[e2, 3])
        m_el[e1, 2] = no1;  m_el[e1, 3] = no2
        m_el[e2, 2] = 0;    m_el[e2, 3] = 0
        m_el[e1, 4] += 1
        m_el[e1, 4] <= MAX_SUB || error("more than $MAX_SUB sub-nodes between two corners")
        m_el[e1, 4 + m_el[e1, 4]] = i
    end

    live_rows = [i for i in 1:nelem if m_el[i, 2] != 0 && m_el[i, 3] != 0]
    nmel = length(live_rows)
    max_sub_cols = maximum(m_el[live_rows, 4]; init = 0)
    m_elem = zeros(Int, nmel, 4 + max_sub_cols)
    for (new_idx, old_idx) in enumerate(live_rows)
        nsub = m_el[old_idx, 4]
        m_elem[new_idx, 1] = new_idx
        m_elem[new_idx, 2] = m_el[old_idx, 2]
        m_elem[new_idx, 3] = m_el[old_idx, 3]
        m_elem[new_idx, 4] = nsub
        nsub > 0 && (m_elem[new_idx, 5:(4 + nsub)] = m_el[old_idx, 5:(4 + nsub)])
    end

    nmno = 0
    m_node_data = Vector{Vector{Float64}}()
    for i in 1:nnode
        node_prop[i, 3] == 0 && continue
        nmno += 1
        node_prop[i, 2] = nmno
        push!(m_node_data, [nmno, node[i, 2], node[i, 3], i, node_prop[i, 3]])
    end
    for i in 1:nnode
        node_prop[i, 3] == 0 && continue
        for j in 1:nmel
            m_elem[j, 2] == i && (m_elem[j, 2] = node_prop[i, 2])
            m_elem[j, 3] == i && (m_elem[j, 3] = node_prop[i, 2])
        end
    end
    for i in 1:nmno, j in 1:nmel
        m_elem[j, 2] == i && push!(m_node_data[i], j)
        m_elem[j, 3] == i && push!(m_node_data[i], -j)
    end
    max_cols = maximum(length.(m_node_data))
    m_node = zeros(Float64, nmno, max_cols)
    for i in 1:nmno
        m_node[i, 1:length(m_node_data[i])] = m_node_data[i]
    end
    nsno = 0
    for i in 1:nnode
        node_prop[i, 3] == 0 || continue
        nsno += 1
        node_prop[i, 2] = nmno + nsno
    end
    return m_node, m_elem, node_prop
end

# ── cFSM mechanical base ──────────────────────────────────────────────────────

"""
Geometry-only (length-independent) cFSM quantities, computed once per
section: the literal (every bend node is a main node) cFSM bookkeeping and
constraint matrices, plus which main nodes are primary in the elastic-corner
model (`corner_model = :elastic`; `:literal` makes every main node primary).
"""
function base_cache(node, elem; angle_tol_deg = 5.0, corner_model = :elastic)
    corner_model in (:elastic, :literal) || error("corner_model must be :elastic or :literal, got $corner_model")
    classes, primary = corner_topology(node, elem; angle_tol_deg)
    corner_model == :literal && (primary = trues(size(node, 1)))
    elprop = cFSM._elemprop(node, elem)
    m_node, m_elem, node_prop = meta_elems_from_classes(node, elem, classes)
    nmno, ncno, nsno = cFSM._node_class(node_prop)
    ndm, nlm = cFSM._mode_nr(nmno, ncno, nsno, m_node)
    DOFperm  = cFSM._DOF_ordering(node_prop)
    Rx, Rz   = cFSM._constr_xz_y(m_node, m_elem)
    Rys      = cFSM._constr_ys_ym(node, m_node, m_elem, node_prop)
    Rud      = cFSM._constr_yu_yd(m_node, m_elem)
    Ryd      = cFSM._constr_yd_yg(node, elem, node_prop, Rys, nmno)
    dy, ngm  = cFSM._y_DOFs(node, elem, m_node, nmno, ndm, Ryd, Rud)
    main_node_idx = [i for i in 1:size(node, 1) if node_prop[i, 4] != 3]   # node order == main-node order
    primary_main  = primary[main_node_idx]
    return (; classes, primary, corner_model, elprop, node_prop, nmno, ncno, nsno, ndm, nlm, DOFperm,
              Rx, Rz, Rys, Ryd, dy, ngm, main_node_idx, primary_main)
end

"""
Full base matrix `B = [R_G R_D R_corner R_L R_O]` (ndof × ndof, CUFSM DOF
order) at half-wavelength `a`, every column normalised per `normalization`,
plus the column ranges of the classes (`ranges.L` spans corner + L; the
corner sub-range is `ranges.corner`). With secondary (non-primary) main
nodes present, the literal G+D Vlasov space is split by static condensation
of the secondary warping DOFs through `Kred = Tᵀ K T` -- see the module
docstring.
"""
function base_vectors(cache, node, elem, prop, a; BC = "S-S", normalization = :vector, K = nothing)
    Rp = cFSM._constr_planar_xz(node, elem, prop, cache.node_prop, cache.DOFperm, 1, a, BC)
    B, ngdm, nlm, _ = cFSM._base_vectors_full(cache.dy, a, 1, elem, cache.elprop, cache.node_prop,
        cache.nmno, cache.ncno, cache.nsno, cache.ngm, cache.ndm, cache.nlm,
        cache.Rx, cache.Rz, Rp, cache.Rys, cache.DOFperm)
    ndof = 4 * size(node, 1)
    size(B) == (ndof, ndof) || error(
        "cFSM base is $(size(B)) but the model has $ndof DOF: G+D+L+O must span the full " *
        "space, which needs a single-branched open section (two free edges).")
    ngm = cache.ngm
    Lcols = (ngdm + 1):(ngdm + nlm)
    Ocols = (ngdm + nlm + 1):ndof

    if all(cache.primary_main)
        ranges = (G = 1:ngm, D = (ngm + 1):ngdm, corner = (ngdm + 1):ngdm, L = Lcols, O = Ocols)
    else
        nV   = ngdm
        BGD  = B[:, 1:nV]
        W    = BGD[2 .* cache.main_node_idx, :]        # main-node warping of every Vlasov column (v DOF = 2i)
        T    = BGD / W                                  # column i: unit warping at main node i, zero elsewhere
        K === nothing && ((K, _) = cFSM._build_K_Kg(node, elem, prop, a, BC, [1]))
        Kred = Symmetric(Matrix(T' * K * T))
        p = findall(cache.primary_main); s = findall(!, cache.primary_main)
        Cmat = zeros(nV, length(p))
        Cmat[p, :] = Matrix{Float64}(I, length(p), length(p))
        Cmat[s, :] = -(Kred[s, s] \ Kred[s, p])         # minimum-energy corner warping for given primaries
        wG = cache.dy[:, 1:ngm]                         # main-node warping of the G modes
        WD = Cmat * nullspace(Matrix(wG' * cache.Ryd * Cmat))     # condensed patterns Ryd-orthogonal to G
        Wc = nullspace(Matrix(hcat(wG, WD)' * cache.Ryd))         # remaining corner-warping directions
        BD = T * WD; BCorner = T * Wc
        nD = size(BD, 2); nC = size(BCorner, 2)
        B = hcat(B[:, 1:ngm], BD, BCorner, B[:, Lcols], B[:, Ocols])
        ranges = (G = 1:ngm, D = (ngm + 1):(ngm + nD), corner = (ngm + nD + 1):(ngm + nD + nC),
                  L = (ngm + nD + 1):(ngm + nD + nC + nlm), O = (ngm + nD + nC + nlm + 1):ndof)
    end

    if normalization == :vector
        for j in axes(B, 2)
            B[:, j] ./= norm(view(B, :, j))
        end
    elseif normalization == :strain_energy
        K === nothing && ((K, _) = cFSM._build_K_Kg(node, elem, prop, a, BC, [1]))
        for j in axes(B, 2)
            b = view(B, :, j)
            B[:, j] ./= sqrt(dot(b, K * b))
        end
    else
        error("normalization must be :vector or :strain_energy, got $normalization")
    end
    return B, ranges
end

_as_matrices(node, elem, prop) = (Matrix{Float64}(node), Matrix{Float64}(elem), Matrix{Float64}(prop))

"Smallest positive load factor of the eigenproblem restricted to the union of `subspaces` (Symbols into `ranges`); NaN if none."
function _constrained_load_factor(B, ranges, subspaces, K, Kg)
    cols = unique(reduce(vcat, (collect(ranges[s]) for s in subspaces)))
    R = B[:, cols]
    Kff  = Matrix(R' * K * R)
    Kgff = Matrix(R' * Kg * R); Kgff = (Kgff + Kgff') / 2
    vals = filter(>(0), real.(eigvals(Kff, Kgff)))
    return isempty(vals) ? NaN : minimum(vals)
end

"""
Smallest positive load factor on the union of `subspaces` together with its
mode vector in the full CUFSM DOF space (d = R c, scaled to unit maximum
component); `(NaN, nothing)` if no positive eigenvalue exists.
"""
function _constrained_mode(B, ranges, subspaces, K, Kg)
    cols = unique(reduce(vcat, (collect(ranges[s]) for s in subspaces)))
    R = B[:, cols]
    Kff  = Matrix(R' * K * R);  Kff  = (Kff + Kff') / 2
    Kgff = Matrix(R' * Kg * R); Kgff = (Kgff + Kgff') / 2
    F = eigen(Kff, Kgff)
    λ = real.(F.values)
    ok = findall(i -> isfinite(λ[i]) && λ[i] > 0 && abs(imag(F.values[i])) <= 1e-8 * max(1.0, abs(λ[i])), eachindex(λ))
    isempty(ok) && return NaN, nothing
    k = ok[argmin(λ[ok])]
    d = R * real.(F.vectors[:, k])
    return λ[k], d ./ maximum(abs, d)
end

_pure_key(s) = Symbol(join(string.(s), "+"))
_as_subspace(s) = s isa Symbol ? [s] : collect(Symbol, s)

# ── Decomposition ─────────────────────────────────────────────────────────────

"""
    decompose_modes(node, elem, prop, lengths; BC = "S-S", angle_tol_deg = 5.0,
                    corner_model = :elastic, normalization = :vector, eig = 1,
                    pure_subspaces = ()) -> DecompositionResult

Runs the conventional (unconstrained) finite strip analysis over `lengths`
on the geometry exactly as given and, at every half-wavelength, decomposes
eigenmode `eig` into its G/D/L/O participations and component shapes -- see
the module docstring for the method, the rounded-corner treatment
(`corner_model`) and the conventions. `node`, `elem`, `prop` are standard
CUFSM arrays.

`pure_subspaces` (e.g. `(:L, :D)` or `(:L, [:G, :D])`) also evaluates the
constrained cFSM curve of each listed subspace at every length, reusing the
base and stiffness already built there; the curves are returned in
`result.pure` keyed `:L`, `:D`, `Symbol("G+D")`, ...
"""
function decompose_modes(node, elem, prop, lengths; BC = "S-S", angle_tol_deg = 5.0,
                         corner_model = :elastic, normalization = :vector, eig = 1,
                         pure_subspaces = ())
    lengths = collect(Float64, lengths)
    node, elem, prop = _as_matrices(node, elem, prop)
    pure_sets = [_as_subspace(s) for s in pure_subspaces]
    pure = Dict{Symbol, Vector{Float64}}(_pure_key(s) => fill(NaN, length(lengths)) for s in pure_sets)
    # Build the cFSM base BEFORE the finite strip run and give CUFSM.strip its
    # own copies: strip rounds the node coordinates it is handed in place, and
    # a base built from the rounded arrays mis-classifies collinear sub-nodes
    # as corners, which silently wrecks the participations.
    cache = base_cache(node, elem; angle_tol_deg, corner_model)
    curve, shapes = CUFSM.strip(prop, copy(node), copy(elem), lengths, [], [], eig)

    nL   = length(lengths)
    load_factor   = zeros(nL)
    participation = zeros(nL, 4)
    corner_part   = zeros(nL)
    coefficients  = Vector{Vector{Float64}}(undef, nL)
    modes         = Vector{Vector{Float64}}(undef, nL)
    components    = Vector{Matrix{Float64}}(undef, nL)
    ranges = nothing

    for (l, a) in enumerate(lengths)
        size(shapes[l], 2) >= eig || error(
            "only $(size(shapes[l], 2)) positive eigenmode(s) at length $a; cannot decompose mode $eig")
        K = Kg = nothing
        isempty(pure_sets) || ((K, Kg) = cFSM._build_K_Kg(node, elem, prop, a, BC, [1]))
        B, ranges = base_vectors(cache, node, elem, prop, a; BC, normalization, K)
        for s in pure_sets
            pure[_pure_key(s)][l] = _constrained_load_factor(B, ranges, s, K, Kg)
        end
        d = shapes[l][:, eig]
        c = B \ d
        class_ranges = (ranges.G, ranges.D, ranges.L, ranges.O)
        comps = hcat((B[:, r] * c[r] for r in class_ranges)...)
        # Participation = 2-norm of each class's component displacement d_M
        # (unique, basis-independent) over the sum of the four. The classic
        # L1-of-coefficients measure is unusable with this base: ~150 nearly
        # parallel L/corner vectors against 4 G vectors inflate whichever
        # class has more columns (a pure global mode read 69 % "Local").
        norms = [norm(view(comps, :, k)) for k in 1:4]
        total = sum(norms)
        participation[l, :] .= 100 .* norms ./ total
        corner_part[l]   = 100 * norm(B[:, ranges.corner] * c[ranges.corner]) / total
        components[l]    = comps
        coefficients[l]  = c
        modes[l]         = d
        load_factor[l]   = curve[l][eig, 2]
    end

    counts = (G = length(ranges.G), D = length(ranges.D), corner = length(ranges.corner),
              L = length(ranges.L), O = length(ranges.O))
    return DecompositionResult(lengths, load_factor, participation, corner_part, coefficients, modes, components,
                               counts, pure, cache.classes, cache.primary, corner_model, node, elem, prop,
                               normalization, eig)
end

"""
    pure_mode_curve(node, elem, prop, lengths, subspaces; BC = "S-S", angle_tol_deg = 5.0,
                    corner_model = :elastic, return_shapes = false)

Constrained (cFSM) signature curve on the geometry exactly as given -- the
eigenproblem restricted to the union of the requested `subspaces` (any of
`:G, :D, :L, :O, :corner`, e.g. `[:L]`, `[:D]`, `[:G, :D]`) of the same
base `decompose_modes` uses. Returns the smallest positive load factor at
each half-wavelength (NaN where none exists); with `return_shapes = true`
returns `(load_factors, shapes)` where `shapes[l]` is the constrained mode
vector in the full CUFSM DOF order (u₁…, v₁…, w₁…, θ₁…; `nothing` where no
positive load factor exists), e.g. for drawing the pure-mode buckled shape.
"""
function pure_mode_curve(node, elem, prop, lengths, subspaces; BC = "S-S", angle_tol_deg = 5.0,
                         corner_model = :elastic, return_shapes = false)
    node, elem, prop = _as_matrices(node, elem, prop)
    cache = base_cache(node, elem; angle_tol_deg, corner_model)
    lf = fill(NaN, length(lengths))
    shapes = Vector{Union{Nothing, Vector{Float64}}}(nothing, length(lengths))
    subs = _as_subspace(subspaces)
    for (l, a) in enumerate(lengths)
        K, Kg = cFSM._build_K_Kg(node, elem, prop, a, BC, [1])
        B, ranges = base_vectors(cache, node, elem, prop, a; BC, K)
        if return_shapes
            lf[l], shapes[l] = _constrained_mode(B, ranges, subs, K, Kg)
        else
            lf[l] = _constrained_load_factor(B, ranges, subs, K, Kg)
        end
    end
    return return_shapes ? (lf, shapes) : lf
end

"""
Locates the signature curve's characteristic minima from the decomposition:
the LOCAL minimum is the L-dominated pointwise trough with the lowest load
factor, the DISTORTIONAL minimum the D-dominated trough with the lowest load
factor ("dominated" = that class has the largest participation there).
Either is `nothing` when no such trough exists in the swept range. Returns
`(local_min, distortional_min, troughs)` as indices into `result.lengths`.
"""
function characteristic_minima(r::DecompositionResult)
    lf = r.load_factor
    troughs = [i for i in 2:(length(lf) - 1) if lf[i-1] > lf[i] < lf[i+1]]
    dominant(i) = argmax(view(r.participation, i, :))      # 1=G 2=D 3=L 4=O
    pick(class_col) = begin
        cands = filter(i -> dominant(i) == class_col, troughs)
        isempty(cands) ? nothing : cands[argmin(lf[cands])]
    end
    return (local_min = pick(3), distortional_min = pick(2), troughs = troughs)
end

# ── Mode-shape geometry ───────────────────────────────────────────────────────

"""
Discretised cross-section (`X, Y`) and in-plane displacement (`ΔX, ΔY`) for a
full-DOF displacement vector `d` (CUFSM DOF order), `n_per_elem` points per
element (shared element end-points de-duplicated), in element order -- which
for a single-branched open section is a single polyline.
"""
function discretized_shape(node, elem, d; n_per_elem = 5)
    n = fill(n_per_elem, size(elem, 1))
    coords, Δ = CUFSMModalGeometry.cross_section_mode_shape_info(elem, node, d, n)
    X = Float64[]; Y = Float64[]; ΔX = Float64[]; ΔY = Float64[]
    ne = length(coords)
    for i in 1:ne
        last = i == ne ? length(coords[i]) : length(coords[i]) - 1
        for j in 1:last
            push!(X, coords[i][j][1]);  push!(Y, coords[i][j][2])
            push!(ΔX, Δ[i][j][1]);     push!(ΔY, Δ[i][j][2])
        end
    end
    return X, Y, ΔX, ΔY
end

"""
Total and G/D/L/O component shapes at length index `l`, all scaled by ONE
common factor so the total's largest in-plane displacement equals 1 -- the
components therefore keep their true relative size (a 5 % class looks small).
Returns `(X, Y, total, G, D, L, O)`, each shape a `(ΔX, ΔY)` tuple.
"""
function component_shapes(r::DecompositionResult, l; n_per_elem = 5)
    X, Y, ΔX, ΔY = discretized_shape(r.node, r.elem, r.shapes[l]; n_per_elem)
    s = 1 / maximum(hypot.(ΔX, ΔY))
    total = (ΔX .* s, ΔY .* s)
    comps = map(1:4) do k
        _, _, cx, cy = discretized_shape(r.node, r.elem, r.components[l][:, k]; n_per_elem)
        (cx .* s, cy .* s)
    end
    return X, Y, total, comps...
end

"Coordinates of the interior primary (corner) nodes -- one per corner arc."
function corner_coordinates(r::DecompositionResult)
    sel = [r.primary[i] && r.classes[i] == 1 for i in eachindex(r.primary)]
    return r.node[sel, 2], r.node[sel, 3]
end

"Per-segment flag (aligned with the `n_per_elem`-discretised polyline) marking reduced/zero-thickness elements, e.g. the zero-thickness hole strip of a net-section model."
function thin_segments(r::DecompositionResult, n_per_elem)
    t = r.elem[:, 4]
    return repeat(t .< 0.99 * maximum(t), inner = n_per_elem - 1)
end

# ── Static figures ────────────────────────────────────────────────────────────

# The figure functions are implemented in ext/BucklingModeIdentificationMakieExt.jl and
# become available when any Makie backend (CairoMakie, GLMakie, ...) is loaded.

"""
    plot_decomposition(r; savepath, title = "", length_unit = "in.", load_label = "Load factor",
                       overlays = [], reference = nothing)

Two linked panels: the conventional signature curve (log-log) with the local
and distortional minima marked, and the stacked G/D/L/O participation (%)
against half-wavelength. `overlays` is a vector of
`(label, lengths, values, color, linestyle)` extra curves (e.g. the pure-L /
pure-D curves from `pure_mode_curve`) and `reference` an optional
`(label, lengths, values)` curve drawn dashed gray.
"""
function plot_decomposition end

"""
    plot_mode_components(r, l; savepath, scale_frac = 0.15, title = "")

One row of five cross-section panels at length index `l`: the total buckled
shape and its G, D, L, O components, drawn on a common scale (total's largest
in-plane displacement = `scale_frac` of the section's largest dimension), so
a class's visual size reflects its participation. The primary corner nodes
used by the decomposition are dotted on the total panel.
"""
function plot_mode_components end

# ── Interactive viewer export ─────────────────────────────────────────────────

_r(v; sig = 5) = round.(Float64.(v); sigdigits = sig)
_rn(v; sig = 5) = [isfinite(x) ? round(Float64(x); sigdigits = sig) : nothing for x in v]   # NaN -> null

"""
    export_viewer_json(r; member, model_label, model_key = "", length_unit, load_unit,
                       overlays = [], reference = nothing, n_per_elem = 3, path = nothing)

Everything the HTML viewer needs, as a JSON string (also written to `path`
when given): the signature curve, the participation at every length, the
undeformed discretised section (with its primary corner nodes and a
per-segment reduced-thickness flag for hole strips) and, per length, the
G/D/L/O component displacement fields on a common scale (total max = 1 --
see `component_shapes`; the viewer rebuilds the total as their sum), plus
the characteristic minima and optional overlay curves. `model_key` is the
short model identifier the multi-profile viewer files data under.
"""
function export_viewer_json(r::DecompositionResult; member, model_label, model_key = "", length_unit = "in.",
                            load_unit = "kip", overlays = [], reference = nothing,
                            n_per_elem = 3, path = nothing)
    mins = characteristic_minima(r)
    X = Y = nothing
    s3(v) = _r(v; sig = 3)
    shapes = map(eachindex(r.lengths)) do l
        Xl, Yl, _, cG, cD, cL, cO = component_shapes(r, l; n_per_elem)
        X, Y = Xl, Yl
        (G = (dx = s3(cG[1]), dy = s3(cG[2])), D = (dx = s3(cD[1]), dy = s3(cD[2])),
         L = (dx = s3(cL[1]), dy = s3(cL[2])), O = (dx = s3(cO[1]), dy = s3(cO[2])))
    end
    cx, cy = corner_coordinates(r)
    data = (
        member = member, model = model_label, model_key = model_key,
        units = (length = length_unit, load = load_unit),
        normalization = String(r.normalization), corner_model = String(r.corner_model), eig = r.eig,
        counts = r.counts,
        lengths = _r(r.lengths), load_factor = _r(r.load_factor),
        participation = (G = _r(r.participation[:, 1]; sig = 4), D = _r(r.participation[:, 2]; sig = 4),
                         L = _r(r.participation[:, 3]; sig = 4), O = _r(r.participation[:, 4]; sig = 4)),
        minima = (local_min = mins.local_min, distortional_min = mins.distortional_min, troughs = mins.troughs),
        geometry = (X = _r(X), Y = _r(Y), thin = collect(Bool, thin_segments(r, n_per_elem))),
        main_nodes = (X = _r(cx), Y = _r(cy)),
        shapes = shapes,
        overlays = [(label = o[1], lengths = _r(o[2]), values = _rn(o[3]), color = string(o[4])) for o in overlays],
        reference = reference === nothing ? nothing :
                    (label = reference[1], lengths = _r(reference[2]), values = _rn(reference[3])),
    )
    json = JSON3.write(data)
    path === nothing || write(path, json)
    return json
end

const _TEMPLATE = joinpath(@__DIR__, "..", "gui", "viewer_template.html")

function _fill_template(template, data_json, manifest_json, title)
    html = read(template, String)
    for ph in ("/*__DATA_JSON__*/", "/*__MANIFEST_JSON__*/", "__TITLE__")
        occursin(ph, html) || error("viewer template has no $ph placeholder")
    end
    return replace(html, "/*__DATA_JSON__*/" => data_json, "/*__MANIFEST_JSON__*/" => manifest_json,
                   "__TITLE__" => title)
end

"""
    write_viewer_html(json, out_path; title = "Mode Decomposition", template = _TEMPLATE)

Writes a self-contained single-profile viewer: the template with its data
placeholder replaced by `json` (no profile menu). Open the result directly in
a browser, or publish it as an artifact.
"""
function write_viewer_html(json::AbstractString, out_path; title = "Mode Decomposition", template = _TEMPLATE)
    write(out_path, _fill_template(template, json, "null", title))
    return out_path
end

"""
    write_profile_js(json, key, path)

Writes one dataset as a script file the multi-profile viewer loads on demand
(`<script src="data/<member>__<model>.js">` works both from disk and when
published alongside the page, unlike `fetch` of a JSON file from disk).
`key` is `"<member>|<model>"`, the viewer's registry key.
"""
function write_profile_js(json::AbstractString, key::AbstractString, path)
    write(path, "window.__MD_REGISTER(" * JSON3.write(key) * ", " * json * ");\n")
    return path
end

"""
    write_multi_viewer_html(out_path; members, models, default_member, default_model, default_json,
                            data_dir = "data", title = "Mode Decomposition", template = _TEMPLATE)

Writes the multi-profile viewer: a profile menu listing `members`, a model
menu listing `models` (vector of `(key, label)`), the default dataset
embedded so the page renders immediately, and every other dataset loaded
from `<data_dir>/<member>__<model>.js` (see `write_profile_js`) on selection.
"""
function write_multi_viewer_html(out_path; members, models, default_member, default_model, default_json,
                                 data_dir = "data", title = "Mode Decomposition", template = _TEMPLATE)
    manifest = JSON3.write((members = collect(String, members),
                            models = [(key = String(k), label = String(lb)) for (k, lb) in models],
                            default = (member = String(default_member), model = String(default_model)),
                            dir = data_dir))
    write(out_path, _fill_template(template, default_json, manifest, title))
    return out_path
end

include("model.jl")
include("api.jl")

end # module
