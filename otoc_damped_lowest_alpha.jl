# Time trace at the LOWEST alpha of each weight-damping kind, head to head.
#
#     julia --project=. otoc_damped_lowest_alpha.jl
#
# otoc_damped_compare.jl's time-trace figure puts every alpha in its own colour
# across two ROWS (Pauli, Majorana) -- the right layout for seeing convergence
# in alpha within a kind, but it buries the Pauli-vs-Majorana comparison at
# fixed (best) alpha across 8 panels. Here there is exactly one curve per kind,
# so kind becomes the colour and site becomes the only panel axis: ONE row, one
# panel per probe site, three lines (Pauli lowest-alpha, Majorana lowest-alpha,
# ED exact).
#
# "Lowest alpha" is taken from whatever damped runs exist at thresh_sel, not
# hardcoded -- so this stays correct as more (finer) alpha runs are added.
#
# From otoc_damped_compare.jl's own accuracy table, lower alpha was the more
# accurate direction for BOTH kinds at every alpha available so far (e.g. Pauli
# alpha=0.25: max|F-ED|=0.319 vs alpha=2: 0.605), so "lowest alpha" is also
# "best available" for these runs -- see that script if a finer alpha changes
# that ordering.

include(joinpath(@__DIR__, "otoc_plotting.jl"))
using LaTeXStrings

# --- What to plot ------------------------------------------------------------
i_site   = 2
pauli    = "ZZ"
lat_tag  = "L243535"
V_sel, J_sel = 0.0, 0.05

thresh_sel = 1e-4          # the family "lowest alpha" is chosen within
min_tmax   = 1.0           # excludes short smoke-test runs
trace_sites = [2, 16, 12, 5]
# -----------------------------------------------------------------------------

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

# ---- collect damped runs at thresh_sel, and the ED reference -----------------
damped = NamedTuple[]
local ed = nothing
for path in find_runs()
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
    length(d["trunc"]) >= 5 || continue
    kind = d["trunc"][5]
    kind in (1.0, 2.0) || continue                       # 0 = plain coefficient clip
    isapprox(d["trunc"][1], thresh_sel; rtol = 1e-6) || continue
    m.tmax >= min_tmax || continue
    push!(damped, (path = path, meta = m, alpha = d["trunc"][4], kind = kind, run = load_run(path)))
end
isempty(damped) && error("no weight-damped runs at thresh=$thresh_sel for V=$V_sel J=$J_sel site $i_site")
ed === nothing && error("no ED reference for V=$V_sel J=$J_sel site $i_site on $lat_tag")

# Lowest alpha PER KIND.
function lowest(kind_code)
    fam = [r for r in damped if r.kind == kind_code]
    isempty(fam) && return nothing
    return fam[argmin([r.alpha for r in fam])]
end
pauli_r, maj_r = lowest(1.0), lowest(2.0)
(pauli_r === nothing && maj_r === nothing) &&
    error("neither damping kind has a run at thresh=$thresh_sel")

println("Lowest alpha per kind at thresh=$(thresh_label(thresh_sel)), V=$V_sel J=$J_sel:")
pauli_r === nothing || println("  Pauli    : α = $(alpha_label(pauli_r.alpha))  ($(basename(pauli_r.path)))")
maj_r    === nothing || println("  Majorana : α = $(alpha_label(maj_r.alpha))  ($(basename(maj_r.path)))")
println("ED reference: $(basename(ed.meta.path))  (t <= $(maximum(ed.time)))")

avail = [j for j in trace_sites if ed.col(j) !== nothing]
bad = setdiff(trace_sites, avail)
isempty(bad) || println("note: ED did not probe site(s) $(join(bad, ", ")); those panels are dropped")

t_win = min(maximum(ed.time),
            minimum(r.run.time[end] for r in (pauli_r, maj_r) if r !== nothing))
ed_i = findall(t -> t <= t_win + 1e-9, ed.time)

# Deviation numbers alongside the figure -- the same accuracy check
# otoc_damped_compare.jl runs, restricted to these two curves.
function dev_vs_ed(r, j)
    c_ed, c_pp = ed.col(j), r.run.col(j)
    (c_ed === nothing || c_pp === nothing) && return NaN
    sh = [(k, findfirst(t -> abs(t - te) < 1e-9, r.run.time)) for (k, te) in enumerate(ed.time)]
    sh = [(k, kp) for (k, kp) in sh if kp !== nothing]
    isempty(sh) && return NaN
    ke = [k for (k, _) in sh]; kp = [k for (_, k) in sh]
    return maximum(abs.(r.run.F[kp, c_pp] .- ed.F[ke, c_ed]))
end
println("-"^62)
@printf("%-8s %6s | %s\n", "kind", "site", "max|F-ED|")
println("-"^62)
for r in (pauli_r, maj_r)
    r === nothing && continue
    kname = r.kind == 2.0 ? "Majorana" : "Pauli"
    for j in avail
        @printf("%-8s %6d | %10.4f\n", kname, j, dev_vs_ed(r, j))
    end
end
println("-"^62)

on_bottom(idx) = true   # one row -- every panel is on the bottom edge
on_left(idx)   = idx == 1

panels = map(enumerate(avail)) do (idx, j)
    p = plot(; base_attrs()...,
             title = "j = $j", titlefontsize = 14, tickfontsize = 12,
             xlabel = "time  t",
             ylabel = on_left(idx) ? L"F_{%$(i_site),j}(t)" : "",
             guidefontsize = 13,
             legend = idx == 1 ? :topright : false, legendfontsize = 10,
             xlims = (0, t_win), grid = true)
    if pauli_r !== nothing
        c = pauli_r.run.col(j)
        if c !== nothing
            ii = findall(t -> t <= t_win + 1e-9, pauli_r.run.time)
            plot!(p, pauli_r.run.time[ii], pauli_r.run.F[ii, c];
                  color = series_color(1), lw = 2,
                  label = idx == 1 ? "Pauli α=$(alpha_label(pauli_r.alpha))" : "")
        end
    end
    if maj_r !== nothing
        c = maj_r.run.col(j)
        if c !== nothing
            ii = findall(t -> t <= t_win + 1e-9, maj_r.run.time)
            plot!(p, maj_r.run.time[ii], maj_r.run.F[ii, c];
                  color = series_color(2), lw = 2,
                  label = idx == 1 ? "Majorana α=$(alpha_label(maj_r.alpha))" : "")
        end
    end
    plot!(p, ed.time[ed_i], ed.F[ed_i, ed.col(j)];
          color = INK_PRIMARY, lw = 2.6, linestyle = :dot,
          label = idx == 1 ? "ED exact" : "")
    p
end

fig = plot(panels...; layout = (1, length(avail)),
           size = (max(480 * length(avail), 1200), 560), dpi = 200,
           plot_title = L"\mathrm{Lowest\!-\!\alpha\ weight\ damping}\ \ F_{%$(i_site),j}(t)\ \ \mathrm{\cdot\ N=16,\ V=%$(V_sel),\ J=%$(J_sel),\ \varepsilon=%$(thresh_label(thresh_sel))}",
           plot_titlefontsize = 17, top_margin = 6Plots.mm,
           left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)

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

out = joinpath(plots_dir(),
    "otoc_damped_lowestalpha_F_N16_$(lat_tag)_$(pauli)_site$(i_site)_V$(V_sel)_J$(J_sel)_eps$(thresh_label(thresh_sel)).png")
s = save_resilient(fig, out)
s === nothing || println("\nSaved: $(basename(s))")
