# import Pkg; Pkg.add(["JSON", "Distributed", "Profile", "Distributions", "Random", "JuMP", "Ipopt", "MadNLP", "LinearAlgebra", "DistributedSparseGrids", "FastGaussQuadrature", "SparseGrids", "Printf", "PrettyTables", "DataFrames"])
_BASE_PATH = @__DIR__

#```
# CMD Line Interface
# ```
using JSON
_PARMS::Dict{String, Any}  = JSON.parse(read(ARGS[1], String))
println("Parameter Values:", _PARMS)
#```

using Distributed
using Profile
using PrettyTables  
import Printf
using DataFrames

if _PARMS["cpu_cores"] < Sys.CPU_THREADS
    addprocs(_PARMS["cpu_cores"])
    println("added $(_PARMS["cpu_cores"]) CPU cores.")
else
    println("Requested $(_PARMS["cpu_cores"]) only $(Sys.CPU_THREADS) available.")
    exit(1)
end

@everywhere begin
    using Distributions
    using Random
    using JuMP
    using Ipopt
    using LinearAlgebra
    using FastGaussQuadrature
    using SparseGrids
end

@everywhere begin
    const _T = 10               # Number of markets
    const _J = 10               # Number of products per market

    const _BETA_MEANS = [1.5, 1.0, 1.0, 1.0]
    const _BETA_SDS   = [1.0, 1.0, 1.0, 0.2]

    const _SIGMA_XW = [0.25 0.05 0.05 0.05;
                       0.05 0.25 0.05 0.05;
                       0.05 0.05 0.1  0.05;
                       0.05 0.05 0.05 0.1 ]
    const _MEAN_XW  = zeros(4)

    const _SIGMA_KSI_OMEGA = [0.2 0.1; 0.1 0.2]
    const _MEAN_KSI_OMEGA = zeros(2)

    const _MC_S_1_000_000 = "Monte Carlo S=1,000,000"
    const _MC_S_100 = "Monte Carlo S=100"
    const _MC_S_1_000 = "Monte Carlo S=1,000"
    const _GH_N_3 = "Gauss-Hermite N=3"
    const _GH_N_5 = "Gauss-Hermite N=5"
    const _SG_A_5 = "Sparse Grid Accuracy=5"

    const _METHODS = [
        (_MC_S_1_000_000, "MT_MONTE_CARLO", $_PARMS["method_parm"][_MC_S_1_000_000]),
        (_MC_S_100, "MT_MONTE_CARLO", $_PARMS["method_parm"][_MC_S_100]),
        (_MC_S_1_000, "MT_MONTE_CARLO", $_PARMS["method_parm"][_MC_S_1_000]),
        (_GH_N_3, "MT_GAUSS_HERMITE", $_PARMS["method_parm"][_GH_N_3]),
        (_GH_N_5, "MT_GAUSS_HERMITE", $_PARMS["method_parm"][_GH_N_5]),
        (_SG_A_5, "MT_SPARSE_GRID", $_PARMS["method_parm"][_SG_A_5])
    ]

    function generate_dataset(;J::Int64 = -1, T::Int64= -1)
        if J == -1 J = _J end
        if T == -1 T = _T end

        x = zeros(T, J, 2)          # Product characteristics
        w = zeros(T, J, 2)          # Cost shifters
        ksi = zeros(T, J)           # Unmeasured demand shocks
        omega = zeros(T, J)         # Unmeasured cost shocks
        mc = zeros(T, J)            # Marginal costs
    
        for t in 1:T
            for j in 1:J
                xw = rand(MvNormal(_MEAN_XW, _SIGMA_XW))
                x[t, j, :] = xw[1:2]
                w[t, j, :] = xw[3:4]
    
                ksi_omega = rand(MvNormal(_MEAN_KSI_OMEGA, _SIGMA_KSI_OMEGA))
                ksi[t, j] = ksi_omega[1]
                omega[t, j] = ksi_omega[2]
    
                log_mc = 0.5 + 0.1 * w[t, j, 1] + 0.1 * w[t, j, 2] + omega[t, j]
                mc[t, j] = exp(log_mc)
            end
        end
        return (x=x, w=w, ksi=ksi, omega=omega, mc=mc)
    end

    function process_method(method_info; J::Int64 = -1, T::Int64= -1, R::Int64 = 1)
        if J == -1 J = _J end
        if T == -1 T = _T end

        method_name, method_type, param = method_info
        println("Solving using $method_name on worker $(myid())...")
        model, fixed_data = create_model(method_type, param)
        prices, time_taken = solve_all_markets(_DATA, model, J=J, T=T)
        return (method_name, prices, time_taken)
    end 

    function create_model(method_type, param; J::Int64 = -1)
        if J == -1 J = _J end

        model = Model(Ipopt.Optimizer)
        set_silent(model)

        @variable(model, p[1:J] >= 0)

        @variable(model, x_k1[1:J])
        @variable(model, x_k2[1:J])
        @variable(model, ksi_k[1:J])
        @variable(model, mc_k[1:J])

        fix.(model[:x_k1], zeros(J))
        fix.(model[:x_k2], zeros(J))
        fix.(model[:ksi_k], zeros(J))
        fix.(model[:mc_k], zeros(J))

        fixed_data = Dict()

        if method_type == "MT_MONTE_CARLO"
            S = param
            beta_i_t = zeros(S, 4)
            for s in 1:S
                beta_i_t[s, :] = [rand(Normal(_BETA_MEANS[k], _BETA_SDS[k])) for k in 1:4]
            end
            beta0_s = beta_i_t[:, 1]
            beta1_s = beta_i_t[:, 2]
            beta2_s = beta_i_t[:, 3]
            beta3_s = abs.(beta_i_t[:, 4])

            fixed_data[:beta0_s] = beta0_s
            fixed_data[:beta1_s] = beta1_s
            fixed_data[:beta2_s] = beta2_s
            fixed_data[:beta3_s] = beta3_s
            fixed_data[:S] = S

            @NLexpression(model, V[s=1:S, k=1:J],
                beta0_s[s] + beta1_s[s] * x_k1[k] + beta2_s[s] * x_k2[k] - beta3_s[s] * p[k] + ksi_k[k])

            @NLexpression(model, exp_V[s=1:S, k=1:J], exp(V[s, k]))

            @NLexpression(model, denom[s=1:S], 1 + sum(exp_V[s, k] for k in 1:J))

            @NLexpression(model, P[s=1:S, k=1:J], exp_V[s, k] / denom[s])

            @NLexpression(model, s_p[j=1:J], (1 / S) * sum(P[s, j] for s in 1:S))

            @NLexpression(model, ds_dp[j=1:J],
                (1 / S) * sum(P[s, j] * (1 - P[s, j]) * (-beta3_s[s]) for s in 1:S))

        else
            nodes, weights = generate_nodes_weights(method_type, param)
            N_nodes = size(nodes, 1)

            beta0_nodes = _BETA_MEANS[1] .+ sqrt(2) * _BETA_SDS[1] .* nodes[:, 1]
            beta1_nodes = _BETA_MEANS[2] .+ sqrt(2) * _BETA_SDS[2] .* nodes[:, 2]
            beta2_nodes = _BETA_MEANS[3] .+ sqrt(2) * _BETA_SDS[3] .* nodes[:, 3]
            beta3_nodes = abs.(_BETA_MEANS[4] .+ sqrt(2) * _BETA_SDS[4] .* nodes[:, 4])

            wts = weights / sum(weights)

            fixed_data[:beta0_nodes] = beta0_nodes
            fixed_data[:beta1_nodes] = beta1_nodes
            fixed_data[:beta2_nodes] = beta2_nodes
            fixed_data[:beta3_nodes] = beta3_nodes
            fixed_data[:wts] = wts
            fixed_data[:N_nodes] = N_nodes

            @NLexpression(model, V[n=1:N_nodes, k=1:J],
                beta0_nodes[n] + beta1_nodes[n] * x_k1[k] + beta2_nodes[n] * x_k2[k] - beta3_nodes[n] * p[k] + ksi_k[k])

            @NLexpression(model, exp_V[n=1:N_nodes, k=1:J], exp(V[n, k]))

            @NLexpression(model, denom[n=1:N_nodes], 1 + sum(exp_V[n, k] for k in 1:J))

            @NLexpression(model, P[n=1:N_nodes, k=1:J], exp_V[n, k] / denom[n])

            @NLexpression(model, s_p[j=1:J], sum(wts[n] * P[n, j] for n in 1:N_nodes))

            @NLexpression(model, ds_dp[j=1:J],
                sum(wts[n] * P[n, j] * (1 - P[n, j]) * (-beta3_nodes[n]) for n in 1:N_nodes))
        end

        @NLconstraint(model, [j=1:J], s_p[j] + (p[j] - mc_k[j]) * ds_dp[j] == 0)
        @NLobjective(model, Min, 0)

        return model, fixed_data
    end

    function solve_all_markets(data, model; J::Int64 = -1, T::Int64= -1)
        if J == -1 J = _J end
        if T == -1 T = _T end

        x = data[:x]
        ksi = data[:ksi]
        mc = data[:mc]
        prices = zeros(T, J)
        total_time = 0.0

        p0 = zeros(J)

        for t in 1:T
            x_t = x[t, :, :]
            ksi_t = ksi[t, :]
            mc_t = mc[t, :]

            fix.(model[:x_k1], x_t[:, 1])
            fix.(model[:x_k2], x_t[:, 2])
            fix.(model[:ksi_k], ksi_t)
            fix.(model[:mc_k], mc_t)

            p0 .= mc_t + rand(J)
            for j in 1:J
                set_start_value(model[:p][j], p0[j])
            end

            start_time = time()
            optimize!(model)
            end_time = time()

            term_status = termination_status(model)
            if term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED
                prices_t = value.(model[:p])
            else
                error("Failed to find equilibrium prices in market $t.")
            end

            prices[t, :] = prices_t
            total_time += end_time - start_time
        end

        return prices, total_time
    end

    function generate_nodes_weights(method_type, param)
        if method_type == "MT_MONTE_CARLO"
            S = param
            beta_i_t = zeros(S, 4)
            for s in 1:S
                beta_i_t[s, :] = [rand(Normal(_BETA_MEANS[k], _BETA_SDS[k])) for k in 1:4]
            end
            return beta_i_t

        elseif method_type == "MT_GAUSS_HERMITE"
            N = param
            nodes_1d, weights_1d = gausshermite(N)
            nodes, weights = multidim_gausshermite(nodes_1d, weights_1d, length(_BETA_MEANS))
            return nodes, weights

        elseif method_type == "MT_SPARSE_GRID"
            level = param
            nodes, weights = smolyak_sparse_grid(level, length(_BETA_MEANS))
            return nodes, weights

        else
            error("Unknown method type")
        end
    end 

    function multidim_gausshermite(nodes_1d, weights_1d, dimension)

        grid = Iterators.product((nodes_1d for _ in 1:dimension)...)
        weights_grid = Iterators.product((weights_1d for _ in 1:dimension)...)
        nodes = collect(grid)
        weights = [prod(w) for w in collect(weights_grid)]
        nodes_matrix =  hcat([collect(node) for node in nodes]...)'
        return nodes_matrix, weights
    end

    function smolyak_sparse_grid(level, dimension)
        nodes, weights = sparsegrid(dimension, level, FastGaussQuadrature.gausshermite)
        return hcat(nodes...)', weights
    end

end  # @everywhere

function print_results(methods_info, times_dict, deviations,
    output_file::String, table_heading::String, append::Bool)

    method_names = []
    parm = []
    times = []
    price_dev = []

    for (method_name, method_type, param) in methods_info
        time_taken = times_dict[method_name]
        push!(method_names, method_name)
        push!(parm, string(param))
        push!(times, round(time_taken, digits=2))
        if method_name != _MC_S_1_000_000
            avg_dev = deviations[method_name]
            push!(price_dev, Printf.format(Printf.Format("%.5f"), avg_dev))
        else
            push!(price_dev, "N/A")
        end
    end
    
    results_table = DataFrame(
        "Method" => method_names,
        "Draws/Nodes" => parm,
        "Time (s)" => times,
        "Avg Price Deviation" => price_dev
    )

    function write_table(io::IO, table::DataFrame, title::String)
        write(io, "\\begin{table}[h]\n")
        write(io, "\\centering\n")
        write(io, "\\caption{" * title * "}\n")
        write(io, "\\resizebox{1\\textwidth}{!}{")
        pretty_table(io, table; backend=Val(:latex))
        write(io, "}\\end{table}\n\n")
    end

    file_mode = append ? "a" : "w"
    open(joinpath(_BASE_PATH, "output", output_file), file_mode) do io
        write_table(io, results_table, table_heading)
    end
    println("\nTables have been saved to '$output_file'")
end

function main()
    Random.seed!(1234)

    data = generate_dataset()
    @everywhere const _DATA = $data

    prices_dict = Dict()
    times_dict = Dict()

    methods_to_process = _METHODS
    results = pmap(process_method, methods_to_process)

    for result in results
        method_name, prices, time_taken = result
        prices_dict[method_name] = prices
        times_dict[method_name] = time_taken
    end

    deviations = Dict()
    for (method_name, _, _) in _METHODS[2:end]
        price_diff = abs.(prices_dict[method_name] .- prices_dict[_MC_S_1_000_000])
        avg_deviation = mean(price_diff)
        deviations[method_name] = avg_deviation
    end
    print_results(_METHODS, times_dict, deviations, "task_2.tex", "Task 2", false)
end

main()
