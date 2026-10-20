# Weight-damped truncation: sweep in alpha, Pauli vs Majorana, against ED.
#
#     julia --project=. otoc_damped_compare.jl
#
# Produces TWO figures:
#   * the truncation sweep  -- C_{i,j}(t) at one probe site, one curve per alpha
#   * the time trace        -- F_{i,j}(t), one panel per probe site
# each with Pauli-damped and Majorana-damped runs side by side and ED in ink.
#
# WHY NOT otoc_truncation_sweep.jl
# --------------------------------
# That script keys its curves on the coefficient threshold, `trunc[1]`, and
# collapses runs sharing one threshold into a single curve ("duplicates, longest
# kept"). Damped runs vary ALPHA at a FIXED threshold, so nine of these eleven
# runs share thresh = 1e-4 and would be reduced to one curve with eight silently
# dropped. It would also place plain coefficient-truncated runs at the same V/J
# on the same axis, which puts three different approximations on one sweep.
# Here the key is (kind, alpha, thresh) and only damped runs are admitted.
#
# READING norm2 HERE -- IT DOES NOT MEAN WHAT IT USUALLY MEANS
# ------------------------------------------------------------
# For coefficient truncation, norm2 is the error diagnostic: the discarded
# weight is the error. For DAMPED truncation that reasoning inverts. A large
# alpha kills high-weight strings on contact, so the operator never spreads; it
# stays compact, and a compact operator RETAINS more of its weight. High norm2
# therefore signals "a small operator was kept", not "the answer is accurate".
# The deviation-from-ED columns below are the only honest accuracy measure, and
# they are printed next to norm2 precisely so the two can be compared.

include(joinpath(@__DIR__, "otoc_plotting.jl"))
using LaTeXStrings

# --- What to plot ------------------------------------------------------------
i_site   = 2        # reference site of W(0)
pauli    = "ZZ"
lat_tag  = "L243535"
V_sel, J_sel = 0.0, 0.05

thresh_sel = 1e-4   # the family to plot; runs at other thresholds are tabled only
min_tmax   = 1.0    # excludes short smoke-test runs

# Renormalized runs (kitaev_OTOC_infinite_temp_normalized.jl) rescale W to unit
# norm every step. That is a DIFFERENT approximation from the ordinary solver, so
# mixing the two on one axis compares two schemes while looking like one sweep.
#   :only    -- renormalized runs only
#   :exclude -- ordinary runs only
#   :any     -- both (they are not distinguished on the plot; use with care)
normalized_sel = :only

sweep_site   = 16                 # probe site for the truncation-sweep figure
trace_sites  = [2, 16, 12, 5]     # panels of the time-trace figure
show_corrected = true             # dashed C/norm2 on the sweep figure
# -----------------------------------------------------------------------------

kind_name(k) = k == 2.0 ? "Majorana" : "Pauli"

function alpha_label(a)
    isapprox(a, round(a); atol = 1e-9) && return string(Int(round(a)))
    return string(round(a, sigdigits = 3))
end

function thresh_label(x::Real)
    e = floor(Int, log10(x)); m = x / 10.0^e
    if m >= 10 - 1e-9; m /= 10; e += 1; end
    ms = isapprox(m, round(m); atol = 1e-6) ? string(Int(round(m))) : string(round(m, sigdigits = 3))
    return "$(ms)e$(e)"
end

# ---- collect the damped runs and the exact reference -------------------------
damped = NamedTuple[]
ed = nothing
for path in find_runs()
    # `ed` is assigned below. Without this, Julia's soft-scope rule makes that
    # assignment create a NEW LOCAL shadowing the global, and the reference on
    # the same line reads as undefined.
    global ed
    local m
    try
        m = parse_meta(path)
    catch
        continue
    end
    (m.lat == lat_tag && m.site == i_site && m.pauli == pauli &&
     isapprox(m.V, V_sel; atol = 1e-12) && isapprox(m.J, J_sel; atol = 1e-12)) || continue
    if m.solver == "ED"
        (ed === nothing || m.tmax > ed.meta.tmax) && (ed = load_run(path))
        continue
    end
    d = npzread(path)
    haskey(d, "trunc") || continue
    length(d["trunc"]) >= 5 || continue          # predates the damping fields
    kind = d["trunc"][5]
    kind in (1.0, 2.0) || continue               # 0 = plain coefficient clip
    m.tmax >= min_tmax || continue               # skip smoke tests
    is_norm = (haskey(d, "normalized") && d["normalized"][1] == 1.0) || haskey(d, "cum_retention")
    (normalized_sel === :any ||
     (normalized_sel === :only    &&  is_norm) ||
     (normalized_sel === :exclude && !is_norm)) || continue
    push!(damped, (path = path, meta = m, thresh = d["trunc"][1],
                   alpha = d["trunc"][4], kind = kind, normalized = is_norm,
                   written = mtime(path), run = load_run(path)))
end
isempty(damped) && error("no weight-damped runs found for V=$V_sel J=$J_sel site $i_site")
ed === nothing && error("no ED reference for V=$V_sel J=$J_sel site $i_site on $lat_tag")

# Dedupe by (kind, thresh, alpha). The same scheme can exist more than once --
# a t_max=3 run and an older t_max=5 run, say -- and both would be drawn under
# one legend label as if they were one curve. Keep the most recently written and
# name the others rather than dropping them silently.
let by_scheme = Dict{Tuple{Float64,Float64,Float64},Vector{NamedTuple}}()
    for r in damped
        push!(get!(by_scheme, (r.kind, r.thresh, r.alpha), NamedTuple[]), r)
    end
    kept, dropped = NamedTuple[], String[]
    for (_, g) in by_scheme
        sort!(g, by = r -> -r.written)
        push!(kept, g[1])
        for r in g[2:end]; push!(dropped, basename(r.path)); end
    end
    global damped = kept
    isempty(dropped) || println("superseded by a newer run of the same scheme (not plotted):\n   " *
                                join(dropped, "\n   "))
end

sort!(damped, by = r -> (r.kind, r.thresh, r.alpha))

# A renormalized run has norm2 == 1, so C/norm2 is identical to C and the dashed
# series would land exactly on the solid one -- implying a correction is being
# shown when none is. Suppress it when every selected run is renormalized.
if all(r -> r.normalized, damped)
    global show_corrected = false
end
# Marks the figures as renormalized so they do not overwrite the ordinary-run
# versions, which are a different approximation at the same V/J/eps.
norm_tag = normalized_sel === :only ? "_norm" :
           normalized_sel === :exclude ? "" : "_mixed"

# One colour per ALPHA, fixed across both figures and both kinds, so the same
# alpha is the same hue everywhere and Pauli/Majorana can be read against each
# other panel to panel.
alphas = sort(unique([r.alpha for r in damped]))
acolor(a) = series_color(findfirst(==(a), alphas))

# ---- inventory + the accuracy table -----------------------------------------
# Deviations are taken only at times BOTH runs have, matched by value: ED is on
# its own output grid and interpolating it would invent reference points.
function dev_vs_ed(r, j)
    c_ed, c_pp = ed.col(j), r.run.col(j)
    (c_ed === nothing || c_pp === nothing) && return (NaN, NaN, NaN)
    sh = [(k, findfirst(t -> abs(t - te) < 1e-9, r.run.time)) for (k, te) in enumerate(ed.time)]
    sh = [(k, kp) for (k, kp) in sh if kp !== nothing]
    isempty(sh) && return (NaN, NaN, NaN)
    ke = [k for (k, _) in sh]; kp = [k for (_, k) in sh]
    Cex = ed.Csq[ke, c_ed]; Fex = ed.F[ke, c_ed]
    Craw = r.run.Csq[kp, c_pp]
    Ccor = Craw ./ r.run.norm2[kp]
    Fpp  = r.run.F[kp, c_pp]
    return (maximum(abs.(Craw .- Cex)), maximum(abs.(Ccor .- Cex)), maximum(abs.(Fpp .- Fex)))
end

println("Weight-damped truncation vs ED   N=$(ed.meta.N) $lat_tag  site $i_site $pauli  " *
        "V=$V_sel J=$J_sel")
println("ED reference: $(basename(ed.meta.path))  (t <= $(maximum(ed.time)))")
println("-"^104)
@printf("%-9s %6s %8s %8s %10s | %12s %12s %12s\n",
        "kind", "alpha", "thresh", "t_max", "norm2_end",
        "max|C-ED|", "max|Ccor-ED|", "max|F-ED|")
println("-"^104)
for r in damped
    dC, dCc, dF = dev_vs_ed(r, sweep_site)
    @printf("%-9s %6s %8s %8.2f %10.5f | %12.4f %12.4f %12.4f\n",
            kind_name(r.kind), alpha_label(r.alpha), thresh_label(r.thresh),
            r.meta.tmax, r.run.norm2[end], dC, dCc, dF)
end
println("-"^104)
println("(deviations at probe site $sweep_site; ED sampling error is ~1.5e-3 on F, ~3e-3 on C)")
println("norm2 RISING with alpha is not accuracy improving -- see the header note.")

# Windows locks a PNG that a viewer has open; one locked file must not abort the
# rest of the figures.
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

plot_family = [r for r in damped if isapprox(r.thresh, thresh_sel; rtol = 1e-6)]
isempty(plot_family) && error("no damped runs at thresh = $thresh_sel")
kinds_present = sort(unique([r.kind for r in plot_family]))

# ================= FIGURE 1: truncation sweep (C vs t) ========================
t_win = min(maximum(ed.time), maximum(r.run.time[end] for r in plot_family))
ed_i  = findall(t -> t <= t_win + 1e-9, ed.time)

sweep_panels = map(kinds_present) do kd
    fam = [r for r in plot_family if r.kind == kd]
    p = plot(; base_attrs()...,
             title = "$(kind_name(kd))-weight damping", titlefontsize = 14, tickfontsize = 11,
             xlabel = "time  t", ylabel = L"C_{%$(i_site),%$(sweep_site)}(t)",
             guidefontsize = 13, legend = :bottomleft, legendfontsize = 9,
             xlims = (0, t_win), grid = true)
    # Style keys carry the SAME linestyle and colour as the curves they label.
    plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2, linestyle = :solid, label = "raw")
    show_corrected && plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2,
                            linestyle = :dash, label = "÷ norm²")
    for r in fam
        c = r.run.col(sweep_site); c === nothing && continue
        idx = findall(t -> t <= t_win + 1e-9, r.run.time)
        hue = acolor(r.alpha)
        show_corrected && plot!(p, r.run.time[idx], r.run.Csq[idx, c] ./ r.run.norm2[idx];
                                color = hue, lw = 1.5, linestyle = :dash, label = "")
        plot!(p, r.run.time[idx], r.run.Csq[idx, c];
              color = hue, lw = 2, label = "α = $(alpha_label(r.alpha))")
    end
    # Exact last => on top, in ink: the reference, not another family member.
    plot!(p, ed.time[ed_i], ed.Csq[ed_i, ed.col(sweep_site)];
          color = INK_PRIMARY, lw = 3, linestyle = :dot, label = "ED exact")
    p
end

fig1 = plot(sweep_panels...; layout = (1, length(sweep_panels)),
            size = (700 * length(sweep_panels), 620), dpi = 200,
            plot_title = L"\mathrm{Weight\!-\!damped\ truncation}\ \ C_{%$(i_site),%$(sweep_site)}(t)\ \ \mathrm{\cdot\ N=16,\ V=%$(V_sel),\ J=%$(J_sel),\ \varepsilon=%$(thresh_label(thresh_sel))}",
            plot_titlefontsize = 17, top_margin = 6Plots.mm,
            left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)
out1 = joinpath(plots_dir(),
    "otoc_damped_sweep_C_N16_$(lat_tag)_$(pauli)_site$(i_site)_to$(sweep_site)" *
    "_V$(V_sel)_J$(J_sel)_eps$(thresh_label(thresh_sel))$(norm_tag).png")
s1 = save_resilient(fig1, out1)
s1 === nothing || println("\nSaved: $(basename(s1))")

# ================= FIGURE 2: time trace (F vs t) =============================
avail = [j for j in trace_sites if ed.col(j) !== nothing]
setdiff(trace_sites, avail) |> bad -> isempty(bad) ||
    println("note: ED did not probe site(s) $(join(bad, ", ")); those panels are dropped")

rows, cols = length(kinds_present), length(avail)
panels = []
for (ri, kd) in enumerate(kinds_present), (ci, j) in enumerate(avail)
    fam = [r for r in plot_family if r.kind == kd]
    idx = (ri - 1) * cols + ci
    p = plot(; base_attrs()...,
             title = ci == 1 ? "$(kind_name(kd))   j = $j" : "j = $j",
             titlefontsize = 13, tickfontsize = 11,
             xlabel = ri == rows ? "time  t" : "",
             ylabel = ci == 1 ? L"F_{%$(i_site),j}(t)" : "",
             guidefontsize = 12,
             legend = idx == 1 ? :topright : false, legendfontsize = 9,
             xlims = (0, t_win), grid = true)
    for r in fam
        c = r.run.col(j); c === nothing && continue
        ii = findall(t -> t <= t_win + 1e-9, r.run.time)
        plot!(p, r.run.time[ii], r.run.F[ii, c];
              color = acolor(r.alpha), lw = 1.8,
              label = idx == 1 ? "α = $(alpha_label(r.alpha))" : "")
    end
    plot!(p, ed.time[ed_i], ed.F[ed_i, ed.col(j)];
          color = INK_PRIMARY, lw = 2.6, linestyle = :dot,
          label = idx == 1 ? "ED exact" : "")
    push!(panels, p)
end

fig2 = plot(panels...; layout = (rows, cols),
            size = (max(430 * cols, 1200), 400 * rows + 90), dpi = 200,
            plot_title = L"\mathrm{Weight\!-\!damped\ truncation}\ \ F_{%$(i_site),j}(t)\ \ \mathrm{\cdot\ N=16,\ V=%$(V_sel),\ J=%$(J_sel),\ \varepsilon=%$(thresh_label(thresh_sel))}",
            plot_titlefontsize = 17, top_margin = 6Plots.mm,
            left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)
out2 = joinpath(plots_dir(),
    "otoc_damped_trace_F_N16_$(lat_tag)_$(pauli)_site$(i_site)" *
    "_V$(V_sel)_J$(J_sel)_eps$(thresh_label(thresh_sel))$(norm_tag).png")
s2 = save_resilient(fig2, out2)
s2 === nothing || println("Saved: $(basename(s2))")
println("\nFigures in $(plots_dir())")
