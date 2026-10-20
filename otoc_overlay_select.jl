# C(t) for an EXPLICIT, HAND-PICKED list of runs, overlaid on one axis.
#
#     julia --project=. otoc_overlay_select.jl
#
# Unlike otoc_damped_compare.jl / otoc_truncation_sweep.jl, which discover runs
# by globbing and then filter them onto a common physics point, this script
# plots exactly the files named in `files` below and nothing else. That is the
# point of it: sometimes the comparison you want is not a clean family.
#
# BECAUSE THE LIST IS HAND-PICKED, NOTHING GUARANTEES THE RUNS ARE COMPARABLE.
# The discovery-based scripts enforce a shared (lattice, V, J, W-site, pauli)
# tuple precisely so a plot cannot put two different physical systems on one
# axis. Here that enforcement is gone, so it is replaced by LOUD REPORTING: the
# script finds the modal value of each parameter across the list, flags every
# run that departs from it, and stamps the departures into the figure's subtitle
# and filename. A mismatch is still plotted -- the user asked for it -- but it
# can never be mistaken for agreement.
#
# WHAT C(t) IS HERE
#   C_ij(t) = Tr |[W_i(t), V_j]|^2 / 2^N = 2*(norm2 - F_ij(t))
# with W_i = Z_i, V_j = Z_j.
#
# THE TWO HORIZONTAL LINES MEAN DIFFERENT THINGS. C = 2 is where the squared
# commutator SETTLES once W(t) has spread (F -> 0 on average), and it is the
# number to read a scrambling plateau against. It is NOT a ceiling: since
# F >= -1, the actual range is
#       0 <= C <= 2*(norm2 + 1)  ->  C <= 4 under exact evolution,
# so an individual C_ij can and does overshoot 2 transiently while F_ij is
# negative, before relaxing back. Only C > 4 is impossible, and that is the
# check this script makes.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

const DIR = joinpath(@__DIR__, "otoc_outputs")

# --- the runs to overlay, exactly as given -----------------------------------
files = [
#  "PP_kitaev_otoc_infT_N16_L243535_wdamp0.1_c1e5_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N16_L243535_mwdamp0.1_c1e5_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "ED_kitaev_otoc_infT_N16_L243535_exact_V0.0_J0.0_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N16_L243535_coeff1e4_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N16_L243535_coeff8e5_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N16_L243535_coeff5e5_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N16_L24e521_coeff1e4_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N16_L24e521_coeff8e5_w32_disc_V0.0_J0.0_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N16_L24e521_coeff5e5_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
  "PP_kitaev_otoc_infT_N32_L48439d_coeff1e4_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
  "PP_kitaev_otoc_infT_N32_L48439d_coeff8e5_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
  "PP_kitaev_otoc_infT_N32_L48439d_coeff5e5_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",

 
]

# One panel per probe site j. Every run here carries all 16 columns, so any
# subset works. j = 2 is the AUTOCORRELATOR for runs seeded at site 2.
probe_sites = [2]

# WHICH QUANTITY: :C for the squared commutator, :F for the OTOC itself.
#   C_ij(t) = 2*(norm2 - F_ij(t))     starts at 0, settles at 2
#   F_ij(t) = Tr[W_i(t) V_j W_i(t) V_j]/2^N    starts at 1, settles at 0
# They carry the SAME information and, under proportional truncation loss, the
# same relative error -- F just looks better because it decays toward 0, so its
# absolute error shrinks with it. Plot both before concluding a scheme is good.
quantity = :F

# WHICH COLUMN each run contributes:
#   :fixed -- every run is read at the panel's probe site j. Only meaningful
#             when the runs share a W site, or the curves are different
#             observables drawn as if they were one.
#   :self  -- every run is read at ITS OWN W site, i.e. the self-correlator
#             (autocorrelator) C_ii / F_ii. This is what makes a run seeded at
#             a different site comparable: the self-correlator is the same
#             function of time at every site related by a lattice symmetry, so
#             W@16's self-correlator and W@2's are the same physical quantity
#             even though the columns differ. `probe_sites` is then ignored and
#             a single panel is drawn.
column_mode = :self

# Add a FRONT-CORRECTED twin for every un-normalized run (see `front_correct` in
# otoc_plotting.jl). The correction restores the weight truncation discarded:
# it lifts C behind the front and lifts F back toward 1 ahead of it, which is
# equivalent to forcing norm2 = 1. Renormalized runs already have norm2 = 1, so
# delta = 0 and a twin would duplicate the original curve exactly -- they are
# skipped rather than drawn twice.
add_front_corrected = true
front_thr = 0.2          # Csq units (the note's C_S threshold of 0.1, doubled)

# HOW THE FRONT IS LOCATED:
#   :vb       -- the note's Eq (14): fit |j-j0| = v_B t_j once from the
#                early-time crossings, then use the straight-line front
#                Theta(v_B t - |j-j0|). Needs the run's lattice to be in
#                KNOWN_LATTICES (otoc_plotting.jl) so distances are available.
#   :per_site -- each site's own threshold crossing, no fit and no distances.
# Both add the same amount (2*l behind the front); they differ only in where the
# front is declared. On this data they agree to ~5e-3; see otoc_vb_front_correction.jl.
front_mode = :vb
# -----------------------------------------------------------------------------

# ---- load --------------------------------------------------------------------
runs = NamedTuple[]
for f in files
    path = joinpath(DIR, f)
    isfile(path) || error("not in otoc_outputs/: $f")
    r = load_run(path)
    d = npzread(path)
    # Renormalized runs are detected from the PAYLOAD, not the name: parse_meta's
    # truncation tag stops at "coeff8e5_w16" and drops the "_norm" segment, so
    # the filename alone would classify a renormalized run as an ordinary one.
    is_norm = (haskey(d, "normalized") && d["normalized"][1] == 1.0) ||
              haskey(d, "cum_retention")
    push!(runs, (meta = r.meta, time = r.time, Csq = r.Csq, F = r.F, norm2 = r.norm2,
                 col = r.col, normalized = is_norm, corrected = false,
                 cum_ret = haskey(d, "cum_retention") ? d["cum_retention"] : nothing))
end

# ---- front-corrected twins ---------------------------------------------------
# Appended as extra runs rather than replacing the originals, so the raw curve
# and its correction appear together and the correction can be judged instead of
# taken on trust.
if add_front_corrected
    twins = NamedTuple[]
    for r in runs
        r.meta.solver == "ED" && continue
        if r.normalized
            println("skipping front-correction of $(r.meta.trunc): renormalized " *
                    "(norm2 == 1, so delta == 0 and the correction is the identity)")
            continue
        end
        # Eq (14) of the lost-weight note: fit ONE butterfly velocity from the
        # early-time crossings, then use the straight-line front
        # Theta(v_B t - |j - j0|). Distances come from the run's OWN lattice
        # fingerprint, never from kitaev_model.jl, which holds only the current
        # lattice and would mislabel every shell of this older data.
        local F_c, C_c
        if front_mode === :vb
            dist = lattice_distances(r.meta.lat, r.meta.site)
            fv = fit_front_velocity(r.time, r.Csq, r.norm2, dist; thr = front_thr)
            isfinite(fv.vB) || error("$(r.meta.trunc): no crossing taken while l was " *
                                     "small enough to fit v_B; lower front_thr or use :per_site")
            @printf("front fit %-22s v_B = %.4f bonds/time  (%d crossings, RMS %.3f bonds)\n",
                    r.meta.trunc, fv.vB, fv.used, fv.resid)
            F_c, C_c, _ = vb_front_correct(r.time, r.F, r.Csq, r.norm2, dist, fv.vB)
        else
            F_c, C_c, _, tj = front_correct(r.F, r.Csq, r.norm2; thr = front_thr)
            never = [j for j in 1:size(r.Csq, 2) if tj[j] > size(r.Csq, 1)]
            isempty(never) || @warn "front never reached thr=$front_thr at site(s) " *
                                    "$(join(never, ", ")) in $(r.meta.trunc); C left uncorrected there"
        end
        # Self-test: either scheme is "force norm2 = 1 behind the front", so the
        # corrected pair must satisfy Csq = 2(1 - F) exactly. A failure means the
        # shared implementation no longer does what this script assumes.
        resid = maximum(abs.(C_c .- 2 .* (1 .- F_c)))
        resid < 1e-9 || error("front correction broke Csq = 2(1-F): residual $resid")
        # norm2 is set to 1: that IS the corrected state, and it keeps the bound
        # check below testing the right inequality for these curves.
        push!(twins, (meta = r.meta, time = r.time, Csq = C_c, F = F_c,
                      norm2 = ones(length(r.time)), col = r.col,
                      normalized = r.normalized, corrected = true, cum_ret = nothing))
    end
    append!(runs, twins)
end

# ---- comparability audit -----------------------------------------------------
# The modal value of each parameter is taken as the intended one; anything else
# is a departure. Reported rather than corrected -- the file list is the user's.
function modal(f)
    vals = [f(r) for r in runs]
    counts = Dict(v => count(==(v), vals) for v in unique(vals))
    return argmax(counts)
end
ref = (lat = modal(r -> r.meta.lat), V = modal(r -> r.meta.V), J = modal(r -> r.meta.J),
       site = modal(r -> r.meta.site), pauli = modal(r -> r.meta.pauli),
       dt = modal(r -> r.meta.dt))

function departures(r)
    d = String[]
    r.meta.lat   == ref.lat   || push!(d, "lattice $(r.meta.lat)")
    r.meta.V     == ref.V     || push!(d, "V=$(r.meta.V)")
    r.meta.J     == ref.J     || push!(d, "J=$(r.meta.J)")
    # Under :self a differing W site is the POINT, not a defect: each run is
    # read at its own site, so the curves are the same observable regardless.
    # Flagging it there would cry wolf on every run that is correctly handled.
    column_mode === :self || r.meta.site == ref.site ||
        push!(d, "W on site $(r.meta.site)")
    r.meta.pauli == ref.pauli || push!(d, r.meta.pauli)
    r.meta.dt    == ref.dt    || push!(d, "dt=$(r.meta.dt)")
    return d
end

println("\n", "="^100)
println("OVERLAY of $(length(runs)) hand-picked run(s)")
println("="^100)
@printf("reference (modal) physics point: lattice %s  V=%s  J=%s  W on site %d  %s  dt=%s\n",
        ref.lat, ref.V, ref.J, ref.site, ref.pauli, ref.dt)
println("-"^100)
qname = string(quantity)
# The quantity actually being plotted. Both arrays are carried so the bound
# checks below can always test C, whichever one the figure shows.
dat(r) = quantity === :F ? r.F : r.Csq

# The column a run contributes to panel j.
colsel(r, j) = column_mode === :self ? r.col(r.meta.site) : r.col(j)

# Panels. Under :self there is one curve per run and nothing to index by probe
# site, so a single panel is drawn whatever `probe_sites` says.
panel_sites = column_mode === :self ? [ref.site] : probe_sites

# Statistics over the columns ACTUALLY PLOTTED, not over all 16. A max taken
# across every column describes a figure this script does not draw.
plotted(r) = [c for c in (colsel(r, j) for j in panel_sites) if c !== nothing]

# Deviation from ED, if an exact run is present -- the number the whole
# comparison exists to produce, so it belongs in the table rather than being
# eyeballed off the curves.
ed_run = findfirst(r -> r.meta.solver == "ED", runs)
function dev_from_ed(r)
    ed_run === nothing && return NaN
    ed = runs[ed_run]
    r === ed && return 0.0
    cols_r, cols_e = plotted(r), plotted(ed)
    (isempty(cols_r) || isempty(cols_e)) && return NaN
    n = min(length(r.time), length(ed.time))
    return maximum(abs.(dat(r)[1:n, cols_r] .- dat(ed)[1:n, cols_e]))
end

@printf("%-26s %6s %6s %5s %8s %9s %10s %10s %11s  %s\n",
        "truncation", "V", "J", "site", "t_max", "norm2_end",
        "$(qname)_end", "max $qname", "max |$qname-ED|", "departures")
println("-"^100)
n_mismatch = 0
for r in runs
    dep = departures(r)
    isempty(dep) || (global n_mismatch += 1)
    cols_r = plotted(r)
    @printf("%-26s %6.3g %6.3g %5d %8.2f %9.5f %10.4f %10.4f %11.4f  %s\n",
            r.meta.trunc * (r.normalized ? "_norm" : "") * (r.corrected ? "+FC" : ""),
            r.meta.V, r.meta.J,
            r.meta.site, r.meta.tmax, r.norm2[end],
            maximum(dat(r)[end, cols_r]), maximum(dat(r)[:, cols_r]),
            dev_from_ed(r),
            isempty(dep) ? "-" : join(dep, ", "))
end
println("-"^100)

if n_mismatch > 0
    println("\n!! $n_mismatch of $(length(runs)) run(s) DEPART from the modal physics point.")
    println("   Curves on one axis that solve different Hamiltonians, or track an")
    println("   operator seeded at a different site, are NOT a truncation comparison.")
    println("   They are plotted as requested and flagged in the figure subtitle.")
end

# The real bound is C <= 2*(norm2 + 1), which is 4 under exact evolution and
# LOWER for a truncated run whose norm2 has decayed. Overshooting 2 is ordinary
# transient behaviour (F_ij < 0) and is not flagged; breaking the run's own
# 2*(norm2+1) is arithmetically impossible and means the file is corrupt or the
# solver is wrong.
# Both checks are run on C whichever quantity is plotted, since C = 2(norm2 - F)
# makes them equivalent statements about the same file:
#     C >= 0            <=>  F <= norm2
#     C <= 2*(norm2+1)  <=>  F >= -1
viol = NamedTuple[]
for r in runs
    cols_r = plotted(r)
    for (k, t) in enumerate(r.time)
        hi = 2 * (r.norm2[k] + 1) + 1e-6
        mx = maximum(r.Csq[k, cols_r]); mn = minimum(r.Csq[k, cols_r])
        if mx > hi
            push!(viol, (r = r, t = t, what = "C=$(round(mx, digits=4)) > 2(norm2+1)=$(round(hi, digits=4))")); break
        elseif mn < -1e-6
            push!(viol, (r = r, t = t, what = "C=$(round(mn, digits=4)) < 0  (F > norm2)")); break
        end
    end
end
if isempty(viol)
    println("\nBound check: every run satisfies 0 <= C <= 2*(norm2+1) at every time,")
    println("equivalently -1 <= F and F <= norm2.")
    quantity === :C &&
        println("(Transient C > 2 is expected wherever F_ij < 0; the plateau, not the peak, is what 2 marks.)")
    quantity === :F &&
        println("(F < 0 is expected transiently -- it is an overshoot past full scrambling, not an error.)")
else
    println("\n!! $(length(viol)) run(s) BREAK a bound that holds by construction:")
    for v in viol
        @printf("     %-26s at t=%.2f: %s\n", v.r.meta.trunc, v.t, v.what)
    end
end

# ---- figure -------------------------------------------------------------------
# ED black (it is the reference, not one series among many); PP runs take series
# slots in the documented order.
function curve_label(r)
    base = r.meta.solver == "ED" ? "ED (exact)" : r.meta.trunc
    extra = String[]
    r.normalized && push!(extra, "renorm.")
    r.corrected && push!(extra, "front correction")
    # No qualifiers -> no parentheses. Appending them unconditionally produced
    # labels like "ED (exact)()" in the legend.
    return isempty(extra) ? base : base * " (" * join(extra, ",  ") * ")"
end

# Colour identifies the RUN; linestyle identifies what was done to it. So a
# front-corrected twin keeps its parent's hue and differs only in style, which
# is what makes the pair readable as "before and after" rather than as two
# unrelated series. The key therefore deliberately ignores `corrected`.
run_key(r) = "$(r.meta.trunc)|$(r.meta.site)|$(r.normalized)"
pp_keys = String[]
for r in runs
    r.meta.solver == "ED" && continue
    k = run_key(r)
    k in pp_keys || push!(pp_keys, k)
end

function curve_style(r)
    r.meta.solver == "ED" && return (color = INK_PRIMARY, lw = 2.6, ls = :solid)
    return (color = series_color(findfirst(==(run_key(r)), pp_keys)),
            lw = r.corrected ? 2.2 : 2,
            ls = r.corrected ? :dot : (r.normalized ? :dash : :solid))
end

styles = map(curve_style, runs)

cols = min(2, length(panel_sites))     # never leave an empty grid slot
rows = ceil(Int, length(panel_sites) / cols)

# The y range is taken FROM THE DATA. A fixed ceiling near 2 silently clipped
# curves that legitimately peak above 3, which is the whole reason the bound is
# documented as 2*(norm2+1) rather than 2.
ymax = maximum(maximum(dat(r)[:, plotted(r)]) for r in runs)
ymin = minimum(minimum(dat(r)[:, plotted(r)]) for r in runs)
pad  = 0.06 * max(ymax - ymin, 0.1)
# Each quantity keeps its own landmark in view: C always shows 0 and the
# saturation line at 2; F always shows its starting value 1 and the scrambled
# value 0, so a curve that never leaves 1 still reads as "never spread".
ylim = quantity === :F ? (min(ymin - pad, -0.05), max(ymax + pad, 1.05)) :
                         (0, max(2.1, ymax + pad))
# Where the quantity settles once W(t) has spread over the lattice.
rule = quantity === :F ? 0.0 : 2.0

# F and C run in OPPOSITE directions: C starts at 0 and climbs to 2, while
# F starts at 1 and falls to 0 (they are the same information, C = 2(norm2-F)).
# Side by side that inversion makes the two figures hard to read against each
# other -- a scheme that under-scrambles sits LOW in C and HIGH in F. Flipping
# F's y axis puts both "more scrambled" directions upward, so the curves have
# the same shape in both plots and can be compared at a glance.
#
# The axis is reversed, NOT the data: tick labels still read 1.0 at the start
# and 0.0 at saturation, so no value is misreported. Set false for a
# conventional F axis.
flip_F = true
yflip_on = (quantity === :F) && flip_F
on_bottom(i) = i > length(panel_sites) - cols
on_left(i) = mod1(i, cols) == 1

panels = map(enumerate(panel_sites)) do (idx, j)
    # A per-panel title earns its place only when several panels must be told
    # apart. With one panel it is drawn inside the multi-line figure title and
    # overlaps it, so the probe is named in the figure title instead.
    p = plot(; base_attrs()...,
             title = length(panel_sites) > 1 ? "probe site j = $j" : "",
             titlefontsize = 12, tickfontsize = 10,
             xlabel = on_bottom(idx) ? "time  t" : "",
             ylabel = on_left(idx) ? (quantity === :F ? L"F_{ij}(t)" : L"C_{ij}(t)") : "",
             guidefontsize = 11,
             # Top-right is the empty quadrant once the curves settle near 2;
             # top-left sat on top of the early peak and hid it.
             legend = idx == 1 ? :topright : false, legendfontsize = 7,
             ylims = ylim, yflip = yflip_on, grid = true)
    # Saturation rule: where C settles once W has spread. Drawn as chrome, not a
    # series. Deliberately NOT treated as a ceiling -- see the header; curves
    # overshoot it transiently and the axis is scaled to show that, not clip it.
    hline!(p, [rule]; color = INK_MUTED, lw = 1, ls = :dot, label = "")
    # ED first, so it lies UNDER the approximate runs. Drawn last it covered
    # them wherever they agree, which reads as the PP curves being absent
    # exactly where they are most accurate.
    for (r, st) in sort(collect(zip(runs, styles)), by = x -> x[1].meta.solver == "ED" ? 0 : 1)
        c = colsel(r, j)
        c === nothing && continue
        plot!(p, r.time, dat(r)[:, c];
              color = st.color, lw = st.lw, linestyle = st.ls,
              label = curve_label(r))
    end
    p
end

# ONE title line, assembled from the data rather than hardcoded so it cannot go
# stale when `files`, `quantity` or `column_mode` change. The descriptors that
# used to sit here -- the rule's value, the probe mode, the meaning of each
# linestyle -- are already carried by the dotted rule itself and by the curve
# labels, so dropping them loses nothing.
qlabel = quantity === :F ? "Infinite temperature OTOC F(t)" : "Squared commutator C(t)"
probe_txt = column_mode === :self ? "$(ref.pauli) self-correlator" :
            "$(ref.pauli) probe j=" * join(panel_sites, ",")
head = "$qlabel | N=$(first(runs).meta.N) Kitaev-Heisenberg PBC Lattice " *
       "at J=$(ref.J), $probe_txt"

# The second line carries only what is true of THIS figure: an inverted axis
# (so its direction cannot be misread) and any departure from the modal physics
# point. Neither is folded into the title line -- appending even one of them
# pushed it off the right edge of the canvas.
notes = String[]
yflip_on && push!(notes, "y axis INVERTED: F falls upward, so F and C read the same way")
n_mismatch == 0 || push!(notes,
    "WARNING: $n_mismatch run(s) depart from V=$(ref.V), J=$(ref.J), W@$(ref.site) - see legend")
sub = isempty(notes) ? "" : "\n" * join(notes, "   |   ")

fig = plot(panels...; layout = (rows, cols),
           size = (max(560 * cols, 1000), max(400 * rows + 150, 640)), dpi = 200,
           plot_title = head * sub,   # `sub` already carries its own newline, or is empty
           plot_titlefontsize = 11, top_margin = 6Plots.mm,
           left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)

# Windows locks a PNG an open image viewer holds; one locked file must not
# discard the whole render.
function save_resilient(fig, path)
    for attempt in 1:3
        try
            savefig(fig, path); return path
        catch
            attempt < 3 && (sleep(1.5); continue)
            alt = replace(path, ".png" => "_new.png")
            try
                savefig(fig, alt)
                @warn "could not overwrite $(basename(path)); wrote $(basename(alt))"
                return alt
            catch err
                @error "could not save $(basename(path))" exception = err
                return nothing
            end
        end
    end
end

# "_mixed" in the name when the list is not a matched comparison, so the file
# cannot later be cited as one.
# The probe set goes in the name: a one-panel j=2 figure and a four-panel
# j=2,5,12,16 figure are different plots, and a shared name would have one
# silently replace the other.
out = joinpath(plots_dir(),
    "otoc_overlay_$(qname)_N$(first(runs).meta.N)_$(ref.lat)_$(ref.pauli)_V$(ref.V)_J$(ref.J)" *
    (column_mode === :self ? "_self" : "_j" * join(panel_sites, "-")) *
    (any(r -> r.corrected, runs) ? "_fc" : "") *
    (n_mismatch > 0 ? "_mixed" : "") * ".png")
s = save_resilient(fig, out)
s === nothing || println("\nSaved: $(basename(s))")
