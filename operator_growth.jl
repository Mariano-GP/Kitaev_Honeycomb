
# Plots |W|(t) — the number of Pauli terms in the evolved operator — for each
# PP config. Data is read from the PP_corr_*.npz files saved by correlation.jl
# (run `julia correlation.jl` first to generate them).

using NPZ
using Plots
using Printf

out_dir   = @__DIR__
plots_dir = joinpath(out_dir, "plots")
mkpath(plots_dir)

N = size(npzread(joinpath(out_dir, "J.npy")), 1)   # derived from current J.npy

configs = [
    #  label            alpha   thresh (ε)
    # ("α=0.05, ε=1e-4",    0.05,   1e-4), 
    # ("α=0.1, ε=1e-4",    0.1,   1e-4), 
    # ("α=0.2, ε=1e-4",    0.2,   1e-4),
    # ("α=0.3, ε=1e-4",    0.3,   1e-4),
    # ("α=0.4, ε=1e-4",    0.4,   1e-4),  
    # ("α=0.5, ε=1e-4",    0.5,   1e-4), 
    # ("α=1.0, ε=1e-4",    1.0,   1e-4), 
	# ("α=2.0, ε=1e-4",    2.0,   1e-4), 
	# ("α=3.0, ε=1e-4",    3.0,   1e-4), 
	# ("α=4.0, ε=1e-4",    4.0,   1e-4), 
	# ("α=5.0, ε=1e-4",    5.0,   1e-4), 
	# ("α=7.0, ε=1e-4",    7.0,   1e-4), 
	# ("α=9.0, ε=1e-4",    9.0,   1e-4),
    ("α=2.0, ε=1e-2",    2.0,   1e-2), 
	("α=2.0, ε=1e-3",    2.0,   1e-3), 
	("α=2.0, ε=1e-4",    2.0,   1e-4), 
	("α=2.0, ε=1e-5",    2.0,   1e-5), 
	("α=2.0, ε=1e-6",    2.0,   1e-6), 
	("α=2.0, ε=1e-7",    2.0,   1e-7), 
	("α=2.0, ε=1e-8",    2.0,   1e-8), 
	("α=2.0, ε=1e-9",    2.0,   1e-9), 

]

function load_terms(alpha, thresh)
    safe_label = "N=$(N)_alpha$(alpha)_thresh$(thresh)"
    path       = joinpath(out_dir, "PP_corr_$(safe_label).npz")
    data       = npzread(path)
    return data["time"], data["terms"]
end

fig = plot(;
    xlabel        = "t",
    ylabel        = "|W| (# Pauli terms)",
    title         = "Operator growth  |W|(t)",
    legend        = :topright,
    grid          = true,
    titlefontsize = 14,
    guidefontsize = 13,
    tickfontsize  = 11,
    legendfontsize= 10,
    size          = (800, 500),
    dpi           = 200)

for (label, alpha, thresh) in configs
    @printf("Loading %s\n", label)
    t, terms = load_terms(alpha, thresh)
    plot!(fig, t, terms; label = label, lw = 2)
end

display(fig)
savefig(fig, joinpath(plots_dir, "Fixed_α_operator_growth_N=$(N).png"))
println("Saved: operator_growth_N=$(N).png")
