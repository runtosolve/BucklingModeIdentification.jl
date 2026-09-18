# Static figures for BucklingModeIdentification, loaded automatically when a Makie
# backend (CairoMakie, GLMakie, ...) is loaded alongside the package.
module BucklingModeIdentificationMakieExt

using Makie
using BucklingModeIdentification
using BucklingModeIdentification: DecompositionResult, characteristic_minima, component_shapes, corner_coordinates,
                         thin_segments, MODE_COLORS, MODE_LABELS
import BucklingModeIdentification: plot_decomposition, plot_mode_components

_plain_ticks(vals) = map(v -> v >= 1 ? string(round(Int, v)) : rstrip(rstrip(string(round(v; digits = 6)), '0'), '.'), vals)

"Draws `xs, ys` as consecutive runs, thin (reduced-thickness) runs with `thin_width`."
function _lines_runs!(ax, xs, ys, thin_seg; color, width, thin_width, linestyle = :solid)
    i = 1; n = length(xs)
    while i <= n - 1
        j = i
        while j <= n - 1 && thin_seg[j] == thin_seg[i]
            j += 1
        end
        lines!(ax, xs[i:j], ys[i:j]; color, linestyle, linewidth = thin_seg[i] ? thin_width : width)
        i = j
    end
end

function plot_decomposition(r::DecompositionResult; savepath, title = "", length_unit = "in.",
                            load_label = "Load factor", overlays = [], reference = nothing)
    mins = characteristic_minima(r)
    fig = Figure(size = (900, 860))
    ax1 = Axis(fig[1, 1]; ylabel = load_label, xscale = log10, yscale = log10, title = title, titlesize = 13,
               xticklabelsvisible = false, ytickformat = _plain_ticks)
    ax2 = Axis(fig[2, 1]; xlabel = "Elastic buckling half-wavelength ($length_unit)",
               ylabel = "Modal participation (%)", xscale = log10, limits = (nothing, (0, 100)),
               xtickformat = _plain_ticks)
    linkxaxes!(ax1, ax2)
    rowsize!(fig.layout, 1, Relative(0.56))

    # y-range follows the conventional curve (and reference) only: the cFSM
    # pure-mode overlays climb orders of magnitude past it at long
    # wavelengths and would squash the curve that matters into a sliver.
    ylo, yhi = extrema(r.load_factor)
    if reference !== nothing
        ylo = min(ylo, minimum(reference[3])); yhi = max(yhi, maximum(reference[3]))
    end
    ylims!(ax1, ylo / 1.6, yhi * 1.6)

    if reference !== nothing
        lines!(ax1, reference[2], reference[3]; color = :gray60, linestyle = :dash, linewidth = 1.5, label = reference[1])
    end
    scatterlines!(ax1, r.lengths, r.load_factor; color = :black, markersize = 4, linewidth = 1.5,
                  label = "conventional FSM (decomposed)")
    for (label, L, v, color, ls) in overlays
        lines!(ax1, L, v; color = color, linestyle = ls, linewidth = 1.5, label = label)
        finite = findall(isfinite, v)
        isempty(finite) && continue
        im = finite[argmin(v[finite])]
        scatter!(ax1, [L[im]], [v[im]]; color = (:white, 0.0), strokecolor = color, strokewidth = 1.5, markersize = 10)
        text!(ax1, L[im], v[im]; text = "min: $(round(v[im], digits = 3)) @ L = $(round(L[im], digits = 2))",
              align = (:right, :top), offset = (-8, -8), fontsize = 9, color = color)
    end
    for (idx, sym, color, mark) in ((mins.local_min, "local", MODE_COLORS.L, :circle),
                                    (mins.distortional_min, "distortional", MODE_COLORS.D, :diamond))
        idx === nothing && continue
        Lm, Rm = r.lengths[idx], r.load_factor[idx]
        scatter!(ax1, [Lm], [Rm]; color = color, marker = mark, markersize = 14, strokecolor = :black, strokewidth = 1)
        text!(ax1, Lm, Rm; text = "$sym min: $(round(Rm, digits = 3)) @ L = $(round(Lm, digits = 2))",
              align = (:left, :bottom), offset = (8, 6), fontsize = 10, color = color)
        vlines!(ax2, [Lm]; color = color, linestyle = :dot, linewidth = 1.5)
    end
    # bottom-left is the one region the curve never visits (short wavelengths sit high)
    axislegend(ax1; position = :lb, framevisible = true, backgroundcolor = (:white, 0.85), labelsize = 9)

    # stacked participation, bottom-up: L, D, G, O
    order  = (3, 2, 1, 4)
    lower  = zeros(length(r.lengths))
    for k in order
        upper = lower .+ r.participation[:, k]
        band!(ax2, r.lengths, lower, upper; color = (MODE_COLORS[k], 0.75), label = MODE_LABELS[k])
        lower = upper
    end
    Legend(fig[3, 1], ax2; orientation = :horizontal, framevisible = false, labelsize = 10,
           tellheight = true, tellwidth = false)
    save(savepath, fig)
    return fig
end

function plot_mode_components(r::DecompositionResult, l; savepath, scale_frac = 0.15, title = "")
    X, Y, total, cG, cD, cL, cO = component_shapes(r, l)
    dim   = max(maximum(X) - minimum(X), maximum(Y) - minimum(Y))
    s     = scale_frac * dim
    pad   = 0.25 * dim
    xl    = (minimum(X) - pad, maximum(X) + pad)
    yl    = (minimum(Y) - pad, maximum(Y) + pad)
    cx, cy = corner_coordinates(r)

    fig = Figure(size = (1500, 420))
    Label(fig[0, 1:5], isempty(title) ? "L = $(round(r.lengths[l], digits = 3)), load factor = $(round(r.load_factor[l], digits = 4))" : title;
          fontsize = 13, font = :bold)
    panels = (("Total buckled shape", total, :black),
              ("Global  $(round(r.participation[l, 1], digits = 1)) %", cG, MODE_COLORS.G),
              ("Distortional  $(round(r.participation[l, 2], digits = 1)) %", cD, MODE_COLORS.D),
              ("Local  $(round(r.participation[l, 3], digits = 1)) %", cL, MODE_COLORS.L),
              ("Other  $(round(r.participation[l, 4], digits = 1)) %", cO, MODE_COLORS.O))
    thin = thin_segments(r, 5)
    for (k, (ttl, (dx, dy), color)) in enumerate(panels)
        ax = Axis(fig[1, k]; title = ttl, titlesize = 12, aspect = DataAspect(), limits = (xl, yl))
        hidedecorations!(ax); hidespines!(ax)
        _lines_runs!(ax, X, Y, thin; color = :black, linestyle = :dash, width = 1, thin_width = 0.4)
        _lines_runs!(ax, X .+ s .* dx, Y .+ s .* dy, thin; color = color, width = 2.5, thin_width = 0.8)
        k == 1 && scatter!(ax, cx, cy; color = :black, markersize = 6)
    end
    save(savepath, fig)
    return fig
end

end # module
