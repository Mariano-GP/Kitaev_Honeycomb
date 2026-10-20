# ED vs Pauli-propagation OTOC time traces -- the OTOC counterpart of
# Correlations/time_trace.jl.
#
#     julia --project=. otoc_time_trace.jl
#
# A grid of small panels, one per probe site j, each showing F_{i,j}(t) from the
# exact solver (ED_OTOC_infinite_temp.jl, solid) overlaid with Pauli propagation
# (kitaev_OTOC_infinite_temp.jl, dashed). This is where PP's error is actually
# judged, so the plot is built to make the comparison honest:
#
#   * ED carries a SHADED ERROR BAND. The exact reference is a stochastic trace
#     estimate with a real standard error (saved as "F_err"), so drawing it as a
#     bare line would overstate it. A PP curve sitting inside the band is not
#     measurably wrong; only excursions outside it are.
#   * Only the times the two runs SHARE are compared, matched by value. PP is
#     usually run to a much larger t_max than ED can afford, and lining the two
#     up by array index instead of by time would quietly compare different
#     instants.
#   * Only sites ED actually probed are drawn. ED often probes a subset, and its
#     columns are indexed by "sites", not by site number.
#
# The runs must share reference site, Pauli type and N. Overlaying different
# reference sites compares physically different quantities, so that is checked
# and refused rather than warned about.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- Edit these to the pair you want to compare --------------------------
# The two runs MAY use different dt, and the script handles that: each solver is
# drawn on its own output grid, and |PP - ED| is evaluated only where both have
# a value. A coarser ED grid loses it no accuracy (Krylov evolution is exact at
# any step size; dt there is purely an output grid, not a Trotter step), so long
# as the coarse dt is a multiple of the fine one -- otherwise the grids share
# only t=0 and there is nothing to compare.
# The "L243535" segment is the LATTICE FINGERPRINT (24 bonds, checksum 3535)
# that both solvers now stamp into every filename -- see lattice_tag() in
# kitaev_model.jl. It exists because N alone does not identify a lattice: an
# open 16-site cluster and a 16-site torus would otherwise produce
# identically-named files, and comparing across that boundary silently yields a
# huge "error" that looks like a solver bug. Change the bond lists and the tag
# changes too, so stale outputs stop matching by name -- intended, not a
# nuisance.
#
# Two ED runs exist for this window: dt0.25 (361 s) and dt0.05 (62089 s, i.e.
# 17.2 h -- 172x the cost for 4.7x the points; see the cost note in
# ED_OTOC_infinite_temp.jl). dt0.05 is the default here because it exists and
# matches PP point for point, but dt0.25 gives the same curve for a fraction of
# the compute if you ever need to regenerate.
# Of the four site-2 PP runs on this lattice, coeff1e4_w16 is the LEAST truncated
# (norm2(t=3) = 0.497, against 0.294 for coeff1e4_w4, 0.305 for both coeff5e4
# runs), so it is the fair one to judge PP by: a w4 cap discards everything above
# Pauli weight 4 on a 16-site lattice, which at late t is most of the operator,
# and its disagreement with ED measures that cap rather than the method. Point
# this at a harsher run deliberately if the question is "how far can truncation
# be pushed" -- just don't read that curve as PP's accuracy.
pp_path = joinpath(@__DIR__, "otoc_outputs",
    "PP_kitaev_otoc_infT_N16_L243535_coeff8e5_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax7.0.npz")
ed_path = joinpath(@__DIR__, "otoc_outputs",
    "ED_kitaev_otoc_infT_N16_L243535_exact_V0.0_J0.05_ZZ_site2_dt0.05_tmax7.0.npz")
quantity = :F            # :F (the OTOC) or :Csq (squared commutator)

# WHICH TARGET SITES j GET A PANEL.
#   nothing   -> every site the ED run probed (the default)
#   [2, 9]    -> just those, in the order given
#
# You can only narrow, never widen: PP computes all N sites, but ED computes
# only the `probe_sites` it was asked for, and there is no ED curve to draw for
# a site it never touched. To plot a site outside ED's list, rerun the ED solver
# with that site included -- edit `probe_sites` in ED_OTOC_infinite_temp.jl's
# run block. (Its cost is linear in the probe count, so adding sites is much
# cheaper than extending t_max, which costs quadratically.) The available list
# is printed below and is named in the error if you ask for a missing site.
show_sites = nothing
# ------------------------------------------------------------------------

pp = load_run(pp_path)
ed = load_run(ed_path)

for (name, f) in (("N", m -> m.N), ("reference site", m -> m.site),
                  ("Pauli", m -> m.pauli), ("V", m -> m.V), ("J", m -> m.J))
    f(pp.meta) == f(ed.meta) || error(
        "PP has $name=$(f(pp.meta)) but ED has $name=$(f(ed.meta)). These are different " *
        "physical quantities; overlaying them would not validate anything. Regenerate one " *
        "run so both agree.")
end
# Same N is NOT the same lattice. Refuse a cross-lattice comparison outright:
# it produces a large, plausible-looking disagreement that reads as a solver
# error. Files predating lattice_tag() carry no fingerprint, so they can only be
# warned about, not checked.
if isempty(pp.meta.lat) || isempty(ed.meta.lat)
    @warn "One of these runs predates the lattice fingerprint, so it cannot be verified to " *
          "be on the same lattice as the other. If the bond lists changed since it was " *
          "generated, the comparison is meaningless -- regenerate it to be sure."
elseif pp.meta.lat != ed.meta.lat
    error("PP was run on lattice $(pp.meta.lat) but ED on $(ed.meta.lat). Same N, different " *
          "bonds -- comparing them yields a large disagreement that is a lattice mismatch, " *
          "not solver error. Regenerate one of them on the current lattice.")
end

m = pp.meta
@printf("PP: %s\nED: %s\n", basename(pp_path), basename(ed_path))
@printf("  N=%d  reference site=%d  pauli=%s  V=%s  J=%s\n", m.N, m.site, m.pauli, m.V, m.J)

# TWO different index sets, because the two solvers are plotted at DIFFERENT
# resolutions on purpose:
#
#   pp_plot_idx -- EVERY PP output time inside the comparison window. PP is
#                  cheap per step, so it typically has many more points (dt=0.05
#                  -> 61 points to t=3) and all of them are real data. Drawing
#                  only the subset ED happens to share would throw most of the
#                  PP curve away and hide any structure between ED samples.
#   shared      -- the times BOTH solvers have, used for every |PP - ED| number.
#                  A deviation can only be computed where both actually have a
#                  value; interpolating ED onto PP's finer grid would invent
#                  reference data that was never computed. When the two runs use
#                  the same dt this is simply every point, and the distinction
#                  below collapses harmlessly.
#
# The window is capped at ED's last time so the axes frame the region where a
# comparison exists -- a PP run to t=5 against ED to t=1 would otherwise squash
# the compared part into the left fifth of every panel.
t_window = min(maximum(pp.time), maximum(ed.time))
pp_plot_idx = findall(tp -> tp <= t_window + 1e-9, pp.time)
ed_plot_idx = findall(te -> te <= t_window + 1e-9, ed.time)

# Match by time VALUE, not index -- see the header.
shared = [(k, findfirst(tp -> abs(tp - te) < 1e-9, pp.time))
          for (k, te) in enumerate(ed.time)]
shared = [(k, kp) for (k, kp) in shared if kp !== nothing]
isempty(shared) && error("PP and ED share no output times (different dt?)")
ed_idx = [k for (k, _) in shared]
pp_idx = [kp for (_, kp) in shared]
t = ed.time[ed_idx]
t_pp = pp.time[pp_plot_idx]
t_ed = ed.time[ed_plot_idx]
@printf("  plotting PP at %d point(s) (dt=%.3g) and ED at %d point(s) (dt=%.3g) over t ∈ [0, %.2f]\n",
        length(t_pp), m.dt, length(t_ed), ed.meta.dt, t_window)
@printf("  |PP - ED| evaluated at the %d shared time(s)\n", length(t))

Qpp = quantity === :F ? pp.F : pp.Csq
Qed = quantity === :F ? ed.F : ed.Csq
Eed = ed.F_err === nothing ? nothing :
      (quantity === :F ? ed.F_err : 2 .* ed.F_err)   # Csq = 2(1-F) => error doubles
qname = quantity === :F ? "F" : "C"

println("  sites available from this ED run: ", join(ed.sites, ", "))
probe_sites = if show_sites === nothing
    ed.sites
else
    missing_sites = setdiff(show_sites, ed.sites)
    isempty(missing_sites) || error(
        "show_sites asks for site(s) $(join(missing_sites, ", ")), which this ED run did not " *
        "probe. It has only $(join(ed.sites, ", ")). Rerun ED_OTOC_infinite_temp.jl with those " *
        "sites added to probe_sites in its run block, then point ed_path at the new file.")
    collect(show_sites)
end
println("  plotting $(length(probe_sites)) panel(s): ", join(probe_sites, ", "))

# ONE shared y-range across every panel -- small multiples are only comparable
# if their scales are identical -- but fitted to the data actually present
# rather than pinned to the theoretical full range. Over a short window F barely
# leaves 1.0, and a hard-coded (-1.15, 1.15) would squash every curve into the
# top sliver of each panel, hiding exactly the ED/PP differences the plot exists
# to show. Padded by 8% so curves don't graze the frame.
#
# Fitted to the sites ACTUALLY PLOTTED, so it must come after the show_sites
# filter above -- otherwise narrowing to a few quiet sites would still leave the
# axes stretched by an excluded site's full swing.
# Over the PLOTTED index sets, not the shared ones: PP's finer grid can excurse
# between ED samples, and those excursions must be inside the frame.
lo = minimum(min(minimum(Qed[ed_plot_idx, ed.col(j)]), minimum(Qpp[pp_plot_idx, pp.col(j)]))
             for j in probe_sites)
hi = maximum(max(maximum(Qed[ed_plot_idx, ed.col(j)]), maximum(Qpp[pp_plot_idx, pp.col(j)]))
             for j in probe_sites)
pad = max(0.08 * (hi - lo), 0.02)
ylims = (lo - pad, hi + pad)

cols = ceil(Int, sqrt(length(probe_sites)))
rows = ceil(Int, length(probe_sites) / cols)

ED_COLOR, PP_COLOR = series_color(1), series_color(2)

# Marker styling depends on whether the two output grids are comparable in
# density. With a coarse ED grid against a fine PP one, size contrast is the
# clearest cue for which is which. With matched grids that same contrast just
# hides PP under ED, so both shrink to equal size and ED switches to an open
# marker (see the plot! calls below).
dense_grids = length(t_ed) > 0.5 * length(t_pp)
ed_ms = dense_grids ? 2.2 : 3.2
pp_ms = dense_grids ? 1.8 : 1.6

# Axis labels go on the OUTER edge of the grid only: every panel shares one x
# scale and one y scale, so repeating the labels on all of them is noise that
# crowds out the data. Bottom-edge panels (those with nothing below them, which
# includes the short last row's neighbours) carry the time label; left-column
# panels carry the quantity label.
# LaTeX is used ONLY for the operator expression itself -- F_{i,j}(t) needs real
# subscripts to read correctly. Every other label (panel titles, x axis, legend)
# stays a plain string: they are ordinary words, and typesetting them in math
# mode only changes the font for no gain.
ylab = quantity === :F ?
    L"\mathrm{OTOC}\quad F_{%$(m.site),j}(t)" :
    L"\mathrm{squared\ commutator}\quad C_{%$(m.site),j}(t)"
on_bottom(idx) = idx > length(probe_sites) - cols
on_left(idx)   = mod1(idx, cols) == 1

subplots = map(enumerate(probe_sites)) do (idx, j)
    ce, cp = ed.col(j), pp.col(j)
    ye = Qed[ed_plot_idx, ce]     # ED on its own (coarse) grid
    yp = Qpp[pp_plot_idx, cp]     # PP on its own (fine) grid -- every PP step
    first_panel = idx == 1

    p = plot(; base_attrs()...,
             title = "j = $j", titlefontsize = 16, tickfontsize = 12,
             xlabel = on_bottom(idx) ? "time  t" : "",
             ylabel = on_left(idx) ? ylab : "",
             guidefontsize = 14,
             legend = first_panel ? :topright : false, legendfontsize = 11,
             ylims = ylims, grid = true)

    # The band is drawn but deliberately UNLABELLED. At N=16 with 8 samples
    # sigma is ~1.5e-3 against a panel spanning ~1.9, so the band is ~1 pixel
    # wide and hides entirely under the 2px ED line -- a legend entry for it
    # would point at something invisible. The magnitude is reported numerically
    # in the summary table at the bottom of this script instead, which is where
    # it can actually be read.
    if Eed !== nothing && all(isfinite, Eed[ed_plot_idx, ce])
        err = Eed[ed_plot_idx, ce]
        plot!(p, t_ed, ye .- err; fillrange = ye .+ err, fillalpha = 0.25,
              fillcolor = ED_COLOR, linealpha = 0, label = "")
    end
    # Each solver is marked at ITS OWN output times, so a marker always means "a
    # value was computed here" and the count differs between the series by
    # design: PP is cheap per step and runs on a fine grid, ED is quadratic in
    # output count and runs on a coarse one. Drawing PP only where ED has a
    # sample would discard most of the PP curve.
    #
    # PP therefore gets many small marks, ED few large ones -- the size contrast
    # doubles as the cue for which grid is which, and keeps the dense PP series
    # from merging into a solid band. Different SHAPES too, because the curves
    # sit almost on top of each other wherever PP is accurate and shape survives
    # that overlap where hue alone would not. Surface-colored rings keep the
    # mark underneath readable where they coincide.
    plot!(p, t_pp, yp; label = first_panel ? "PP (dt=$(m.dt))" : "",
          color = PP_COLOR, lw = 1.4, linestyle = :dash,
          marker = :diamond, markersize = pp_ms, markerstrokewidth = 0)
    # ED is drawn second, hence on top. When the grids are comparable in density
    # its markers are OPEN (surface fill, colored ring) so the PP mark
    # underneath stays visible wherever the two coincide -- which, when PP is
    # accurate, is nearly everywhere. A solid ED marker at matched resolution
    # simply erases the PP series from the plot.
    plot!(p, t_ed, ye; label = first_panel ? "ED exact (dt=$(ed.meta.dt))" : "",
          color = ED_COLOR, lw = 2,
          marker = :circle, markersize = ed_ms,
          markercolor = dense_grids ? SURFACE : ED_COLOR,
          # Thin ring: at these small sizes a thicker stroke fills the open
          # marker solid and hides the PP mark underneath again.
          markerstrokewidth = dense_grids ? 0.8 : 0.6,
          markerstrokecolor = dense_grids ? ED_COLOR : SURFACE)
    p
end

fig = plot(subplots...; layout = (rows, cols),
           size = (max(650 * cols, 1240), 460 * rows + 90), dpi = 200,
           plot_title = L"\mathrm{%$(m.pauli)\ OTOC}\ \ %$(qname)_{%$(m.site),j}(t)\ \ \mathrm{-\ ED\ vs\ PP\ \cdot\ N=%$(m.N)\ V=%$(m.V)\ J=%$(m.J)}, ϵ=$(cutoff_label(pp.cut)) ",
           plot_titlefontsize = 20, top_margin = 7Plots.mm,
           left_margin = 12Plots.mm, bottom_margin = 10Plots.mm,
           right_margin = 6Plots.mm)
display(fig)
out = joinpath(plots_dir(), "otoc_ED_vs_PP_$(quantity)_" *
                            replace(basename(pp_path), ".npz" => "", "PP_" => "") * ".png")  
savefig(fig, out)
println("\nSaved: $out")

# The numbers behind the picture. A deviation is only meaningful relative to the
# ED sampling error, so both are reported side by side -- and "within 1σ" is
# stated outright rather than left to be squinted at off the panels.
println("\n" * "-"^70)
@printf("%6s  %14s  %14s  %10s\n", "site", "max|PP-ED|", "ED σ (max)", "verdict")
println("-"^70)
worst_overall = 0.0
for j in probe_sites
    ce, cp = ed.col(j), pp.col(j)
    dev = maximum(abs.(Qed[ed_idx, ce] .- Qpp[pp_idx, cp]))
    sig = Eed === nothing ? NaN : maximum(Eed[ed_idx, ce])
    global worst_overall = max(worst_overall, dev)
    @printf("%6d  %14.3e  %14.3e  %10s\n", j, dev, sig,
            isnan(sig) ? "-" : dev <= sig ? "within 1σ" : "resolved")
end
println("-"^70)
@printf("worst deviation over all probes: %.3e\n", worst_overall)
if pp.norm2[end] > 0.999
    println("PP norm2 = $(round(pp.norm2[end], digits=8)) — truncation is NOT the limiting\n" *
            "error here; a resolved deviation is Trotter error at dt=$(m.dt).")
else
    @printf("PP norm2 = %.6f — truncation has discarded %.2f%% of the operator.\n",
            pp.norm2[end], 100 * (1 - pp.norm2[end]))
end
