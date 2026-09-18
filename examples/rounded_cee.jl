# From corner points to a full modal decomposition of a rounded-corner lipped
# Cee, in four calls. Run with a project that has BucklingModeIdentification (and,
# for the figures, CairoMakie):
#     julia --project=. examples/rounded_cee.jl
using BucklingModeIdentification
using CairoMakie          # optional: enables the static figures

# Lipped Cee, inches: 2.5 deep, 1.5 flanges, 0.5 lips, centerline corner
# radius 0.1, t = 0.0451, E = 29500 ksi
corners = [(1.5, 0.5), (1.5, 0.0), (0.0, 0.0), (0.0, 2.5), (1.5, 2.5), (1.5, 2.0)]
X, Y = polyline_with_fillets(corners, 0.1; n_arc = 4, n_flat = 6)
t, E = 0.0451, 29500.0

axial   = decompose_section(X, Y, t, E; name = "Cee-2.5x1.5", load = :axial)
bending = decompose_section(X, Y, t, E; name = "Cee-2.5x1.5", load = :bending)
holed   = decompose_section(X, Y, t, E; name = "Cee-2.5x1.5", load = :axial, hole = 1.5)   # 1.5 in. web hole

summarize(axial); summarize(bending); summarize(holed)

outdir = joinpath(@__DIR__, "output")
write_outputs(axial;   outdir)     # figures (with CairoMakie), JSON, single-case viewer
write_outputs(bending; outdir)
write_outputs(holed;   outdir)
viewer = build_viewer([axial, bending, holed]; outdir, title = "Cee-2.5x1.5 Mode Decomposition")
println("open in a browser: ", viewer)
