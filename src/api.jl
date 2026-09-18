# High-level API: from a profile's centerline and details straight to the
# decomposition, its summary, figures and viewers.

"One analysed section/load case: the decomposition plus everything needed to label and export it."
struct SectionCase
    name::String
    model_key::String
    model_label::String
    length_unit::String
    load_unit::String
    result::DecompositionResult
    minima::NamedTuple
    overlays::Vector{Any}
end

function _load_case(load, P, Mxx)
    load === nothing && ((P == 0 && Mxx == 0) ? error("give load = :axial or :bending, or nonzero P / Mxx") : return (Float64(P), Float64(Mxx)))
    load === :axial   && return (1.0, 0.0)
    load === :bending && return (0.0, 1.0)
    error("load must be :axial or :bending (or pass P / Mxx explicitly)")
end

"""
    decompose_section(X, Y, t, E; name = "section", load = :axial, hole = nothing, kwargs...) -> SectionCase

Complete analysis of one open thin-walled section from its centerline:

- `X, Y`: node coordinates in order along the section (element i joins nodes
  i, i+1); rounded corners as discretised arcs, ribs, jogs -- the geometry is
  analysed exactly as given, apart from `drop_short_segments` mesh hygiene
  (strips shorter than `min_length_frac * t` are removed).
- `t`, `E`: thickness and modulus in consistent units (`length_unit`, and
  `E` in force/length²).
- `load`: `:axial` (unit compression, load factors in force) or `:bending`
  (unit strong-axis moment, load factors in force·length); or pass `P`/`Mxx`
  explicitly with `load = nothing`.
- `hole`: `nothing`, a hole width (centred on mid-depth), or `(y_lo, y_hi)`;
  builds the zero-thickness net-section model (`zero_thickness_hole`).
- `lengths`: half-wavelengths to sweep; default 70 log-spaced points from
  0.1 to 200 times the section depth.
- `corner_model`, `angle_tol_deg`, `pure_subspaces` as in `decompose_modes`.
- `model_key` / `model_label` name the case (defaults such as `"Pcr_gross"`,
  `"Mcr_hole"`); `load_unit` defaults to `"kip"` / `"kip-in."`.

Returns a `SectionCase` -- see `summarize`, `write_outputs`, `build_viewer`.
"""
function decompose_section(X, Y, t, E; name = "section", load = :axial, P = 0.0, Mxx = 0.0,
                           hole = nothing, lengths = nothing, n_lengths = 70,
                           corner_model = :elastic, angle_tol_deg = 5.0, min_length_frac = 0.1, ν = 0.3,
                           pure_subspaces = (:L, :D), model_key = nothing, model_label = nothing,
                           length_unit = "in.", load_unit = nothing)
    P, Mxx = _load_case(load, P, Mxx)
    Xc, Yc = drop_short_segments(X, Y; min_length = min_length_frac * t)
    if hole === nothing
        t_vec = t
    else
        y_lo, y_hi = hole isa Number ? (mid = (minimum(Yc) + maximum(Yc)) / 2; (mid - hole / 2, mid + hole / 2)) :
                                       (Float64(hole[1]), Float64(hole[2]))
        Xc, Yc, t_vec = zero_thickness_hole(Xc, Yc, t, y_lo, y_hi; tol = min_length_frac * t)
    end
    prop, node, elem = cufsm_model(Xc, Yc, t_vec, E; ν, P, Mxx)

    depth = maximum(Yc) - minimum(Yc)
    Ls = lengths === nothing ? 10 .^ range(log10(0.1 * depth), log10(200 * depth), length = n_lengths) :
                               collect(Float64, lengths)
    r = decompose_modes(node, elem, prop, Ls; corner_model, angle_tol_deg, pure_subspaces)

    axial = Mxx == 0
    key   = model_key === nothing ? (axial ? "Pcr" : "Mcr") * (hole === nothing ? "_gross" : "_hole") : String(model_key)
    label = model_label === nothing ?
            (axial ? "axial compression" : "strong-axis bending") * (hole === nothing ? ", gross section" : ", zero-thickness net section (web hole)") :
            String(model_label)
    lu = load_unit === nothing ? (axial ? "kip" : "kip-in.") : String(load_unit)
    pretty = Dict(:L => "cFSM pure local (L only)", :D => "cFSM pure distortional (D only)", :G => "cFSM pure global (G only)")
    colors = Dict(:L => :dodgerblue, :D => :darkorange, :G => :gray40)
    overlays = Any[(get(pretty, k, "cFSM pure $k"), Ls, v, get(colors, k, :purple), :dash) for (k, v) in r.pure]
    sort!(overlays; by = o -> o[1])
    return SectionCase(String(name), key, label, String(length_unit), lu, r, characteristic_minima(r), overlays)
end

"Prints the characteristic points of a case: the L- and D-dominated troughs with their participations, and the pure-mode minima."
function summarize(c::SectionCase; io = stdout)
    r, m = c.result, c.minima
    println(io, "$(c.name) — $(c.model_label)  [$(c.load_unit)]")
    println(io, "  nodes $(size(r.node, 1)), subspaces G=$(r.counts.G) D=$(r.counts.D) L=$(r.counts.L) O=$(r.counts.O), corner model $(r.corner_model)")
    for (label, idx) in (("local min", m.local_min), ("distortional min", m.distortional_min))
        if idx === nothing
            println(io, "  $(rpad(label, 17)) none (no $(label[1] == 'l' ? "L" : "D")-dominated trough in the swept range)")
        else
            p = r.participation[idx, :]
            println(io, "  $(rpad(label, 17)) L = $(round(r.lengths[idx], digits = 3)) $(c.length_unit)   cr = $(round(r.load_factor[idx], digits = 4)) $(c.load_unit)   ",
                        "G $(round(p[1], digits = 1)) %  D $(round(p[2], digits = 1)) %  L $(round(p[3], digits = 1)) %  O $(round(p[4], digits = 1)) %")
        end
    end
    for (k, v) in sort(collect(r.pure); by = first)
        f = findall(isfinite, v)
        isempty(f) && continue
        i = f[argmin(v[f])]
        println(io, "  pure $(rpad(string(k), 12)) min = $(round(v[i], digits = 4)) $(c.load_unit) @ L = $(round(r.lengths[i], digits = 3)) $(c.length_unit)")
    end
    return nothing
end

"True when a Makie backend is loaded, so `plot_decomposition` / `plot_mode_components` have methods."
plotting_available() = !isempty(methods(plot_decomposition))

"""
    write_outputs(c::SectionCase; outdir, figures = true, viewer = true) -> NamedTuple of paths

Writes, under `outdir` with the prefix `<name>__<model_key>`: the
signature-curve + participation figure and the component-shape figures at
the local and distortional troughs (when a Makie backend is loaded), the
viewer JSON, and a self-contained single-case interactive viewer HTML.
"""
function write_outputs(c::SectionCase; outdir, figures = true, viewer = true)
    mkpath(outdir)
    prefix = "$(c.name)__$(c.model_key)"
    r, m = c.result, c.minima
    paths = Dict{Symbol, String}()
    if figures
        if plotting_available()
            sym = startswith(c.model_key, "P") ? "P" : "M"
            paths[:signature] = joinpath(outdir, "$(prefix)_signature_curve_decomposition.png")
            plot_decomposition(r; savepath = paths[:signature],
                title = "$(c.name) — $(c.model_label) — conventional FSM with G/D/L/O modal participation",
                length_unit = c.length_unit, load_label = "$(sym)cr ($(c.load_unit))", overlays = c.overlays)
            if m.local_min !== nothing
                paths[:local_shapes] = joinpath(outdir, "$(prefix)_mode_components_local_min.png")
                plot_mode_components(r, m.local_min; savepath = paths[:local_shapes])
            end
            if m.distortional_min !== nothing
                paths[:distortional_shapes] = joinpath(outdir, "$(prefix)_mode_components_distortional_min.png")
                plot_mode_components(r, m.distortional_min; savepath = paths[:distortional_shapes])
            end
        else
            @info "No Makie backend loaded -- skipping figures (`using CairoMakie` before calling write_outputs to get them)."
        end
    end
    paths[:json] = joinpath(outdir, "$(prefix)_decomposition.json")
    json = export_viewer_json(r; member = c.name, model_label = c.model_label, model_key = c.model_key,
                              length_unit = c.length_unit, load_unit = c.load_unit, overlays = c.overlays,
                              path = paths[:json])
    if viewer
        paths[:viewer] = joinpath(outdir, "$(prefix)_viewer.html")
        write_viewer_html(json, paths[:viewer]; title = "$(c.name) $(c.model_key) Mode Decomposition")
    end
    return (; paths...)
end

"""
    build_viewer(cases::Vector{SectionCase}; outdir, title = "Mode Decomposition",
                 default = cases[1], filename = "mode_decomposition_all.html") -> path

One interactive viewer for many cases, with a profile menu (case names) and
a model menu (case model keys); the default case is embedded so the page
renders immediately, the others are loaded from `<outdir>/data/` on
selection. Works opened from disk or published with the `data/` files beside
it.
"""
function build_viewer(cases::AbstractVector{SectionCase}; outdir, title = "Mode Decomposition",
                      default = first(cases), filename = "mode_decomposition_all.html")
    data_dir = joinpath(outdir, "data")
    mkpath(data_dir)
    members = String[]; models = Tuple{String, String}[]
    default_json = ""
    for c in cases
        json = export_viewer_json(c.result; member = c.name, model_label = c.model_label, model_key = c.model_key,
                                  length_unit = c.length_unit, load_unit = c.load_unit, overlays = c.overlays)
        write_profile_js(json, "$(c.name)|$(c.model_key)", joinpath(data_dir, "$(c.name)__$(c.model_key).js"))
        c.name in members || push!(members, c.name)
        any(m -> m[1] == c.model_key, models) || push!(models, (c.model_key, c.model_label))
        c === default && (default_json = json)
    end
    isempty(default_json) && error("`default` must be one of `cases`")
    return write_multi_viewer_html(joinpath(outdir, filename); members, models,
                                   default_member = default.name, default_model = default.model_key,
                                   default_json, title)
end
