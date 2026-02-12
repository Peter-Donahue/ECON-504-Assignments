# import Pkg; Pkg.add(["Distributions", "Random", "JuMP", "Ipopt", "DataFrames", "PrettyTables"])

using Distributions
using Random
using JuMP
using Ipopt
using DataFrames
using PrettyTables

_BASE_PATH = @__DIR__

const _T = 10     # Number of markets
const _J = 10     # Number of products per market
const _S = 100    # Number of simulation draws per market

const _BETA_MEANS = [1.5, 1.0, 1.0, 1.0]
const _BETA_SDS = [1.0, 1.0, 1.0, 0.2]

const _SIGMA_XW = [0.25 0.05 0.05 0.05;
                   0.05 0.25 0.05 0.05;
                   0.05 0.05 0.1  0.05;
                   0.05 0.05 0.05 0.1 ]
const _MEAN_XW = zeros(4)

const _SIGMA_KSI_OMEGA = [0.2 0.1; 0.1 0.2]
const _MEAN_KSI_OMEGA = zeros(2)

function main()
    Random.seed!(1234)

    x = zeros(_T, _J, 2)          # Product characteristics
    w = zeros(_T, _J, 2)          # Cost shifters
    ksi = zeros(_T, _J)           # Unmeasured demand shocks
    omega = zeros(_T, _J)         # Unmeasured cost shocks
    mc = zeros(_T, _J)            # Marginal costs
    beta_i = zeros(_T, _S, 4)     # Random coefficients
    prices = zeros(_T, _J)        # Equilibrium prices
    market_shares = zeros(_T, _J) # Market shares

    for t in 1:_T
        x_t, w_t, ksi_t, omega_t, mc_t = generate_market_data()
        x[t, :, :] = x_t
        w[t, :, :] = w_t
        ksi[t, :] = ksi_t
        omega[t, :] = omega_t
        mc[t, :] = mc_t
        beta_i_t = generate_random_coefficients()
        beta_i[t, :, :] = beta_i_t
        prices_t, s_p_t = solve_equilibrium(t, x_t, ksi_t, mc_t, beta_i_t)
        prices[t, :] = prices_t
        market_shares[t, :] = s_p_t
    end
    output_all_data( x, ksi, mc, prices, market_shares)
end

function generate_market_data()
    x_t = zeros(_J, 2)
    w_t = zeros(_J, 2)
    ksi_t = zeros(_J)
    omega_t = zeros(_J)
    mc_t = zeros(_J)

    for j in 1:_J
        xw = rand(MvNormal(_MEAN_XW, _SIGMA_XW))
        x_t[j, :] = xw[1:2]
        w_t[j, :] = xw[3:4]

        ksi_omega = rand(MvNormal(_MEAN_KSI_OMEGA, _SIGMA_KSI_OMEGA))
        ksi_t[j] = ksi_omega[1]
        omega_t[j] = ksi_omega[2]

        log_mc = 0.5 + 0.1 * w_t[j, 1] + 0.1 * w_t[j, 2] + omega_t[j]
        mc_t[j] = exp(log_mc)
    end

    return x_t, w_t, ksi_t, omega_t, mc_t
end

function generate_random_coefficients()
    beta_i_t = zeros(_S, 4)
    for s in 1:_S
        beta_i_t[s, :] = [rand(Normal(_BETA_MEANS[k], _BETA_SDS[k])) for k in 1:4]
    end
    return beta_i_t
end

function solve_equilibrium(t, x_t, ksi_t, mc_t, beta_i_t)
    prices_t = nothing
    converged = false
    for attempt in 1:20
        model = Model(Ipopt.Optimizer)
        set_silent(model)  
        @variable(model, p[1:_J] >= 0)

        beta0_s = beta_i_t[:, 1]
        beta1_s = beta_i_t[:, 2]
        beta2_s = beta_i_t[:, 3]
        beta3_s = abs.(beta_i_t[:, 4]) 

        x_k1 = x_t[:, 1]
        x_k2 = x_t[:, 2]
        ksi_k = ksi_t
        mc_k = mc_t

        @NLexpression(model, V[s=1:_S, k=1:_J], beta0_s[s] + beta1_s[s] * x_k1[k] + beta2_s[s] * x_k2[k] - beta3_s[s] * p[k] + ksi_k[k])
        @NLexpression(model, exp_V[s=1:_S, k=1:_J], exp(V[s, k]))
        @NLexpression(model, denom[s=1:_S], 1 + sum(exp_V[s, k] for k in 1:_J))
        @NLexpression(model, P[s=1:_S, k=1:_J], exp_V[s, k] / denom[s])
        @NLexpression(model, s_p[j=1:_J], (1 / _S) * sum(P[s, j] for s in 1:_S))
        @NLexpression(model, ds_dp[j=1:_J], (1 / _S) * sum(P[s, j] * (1 - P[s, j]) * (-beta3_s[s]) for s in 1:_S))
        @NLconstraint(model, [j=1:_J], s_p[j] + (p[j] - mc_k[j]) * ds_dp[j] == 0)
        @NLobjective(model, Min, 0)

        # Random starting prices above marginal cost
        p0 = mc_k + rand(_J)  
        for j in 1:_J
            set_start_value(p[j], p0[j])
        end
        
        optimize!(model)
        termination_status = JuMP.termination_status(model)
        if termination_status == MOI.OPTIMAL || termination_status == MOI.LOCALLY_SOLVED
            prices_t = value.(p)
            converged = true
            break
        end
    end
    if !converged
        error("Failed to find equilibrium prices in market $t after 20 attempts.")
    end

    s_p = zeros(_J)
    for j in 1:_J
        s_p_j = 0.0
        for s in 1:_S
            beta0 = beta_i_t[s, 1]
            beta1 = beta_i_t[s, 2]
            beta2 = beta_i_t[s, 3]
            beta3 = abs(beta_i_t[s, 4])
            V = beta0 .+ beta1 .* x_t[:, 1] .+ beta2 .* x_t[:, 2] .- beta3 .* prices_t .+ ksi_t
            denom = 1 + sum(exp.(V))
            P_i = exp.(V) ./ denom
            s_p_j += P_i[j] / _S
        end
        s_p[j] = s_p_j
    end
    return prices_t, s_p
end


function output_all_data(x, ksi, mc, prices, market_shares)
    
    output_file = joinpath(_BASE_PATH, "output", "task_1_ext.tex")
    
    all_markets::Vector{Int64} = zeros(_T * _J)
    all_products::Vector{Int64} = zeros(_T * _J)
    all_prices::Vector{Float64} = zeros(_T * _J)
    all_mc::Vector{Float64} = zeros(_T * _J)
    all_ms::Vector{Float64} = zeros(_T * _J)
    all_char1::Vector{Float64} = zeros(_T * _J)
    all_char2::Vector{Float64} = zeros(_T * _J)
    profit_margins::Vector{Float64} = zeros(_T * _J)
    i = 1
    for t in 1:_T
        for j in 1:_J
            all_markets[i] = t
            all_products[i] = j
            all_prices[i] = round(prices[t, j], digits=2)
            all_mc[i] = round(mc[t, j], digits=2)
            all_ms[i] = round(market_shares[t, j], digits=2)
            all_char1[i] = round(x[t, j, 1],digits=2)
            all_char2[i] = round(x[t, j, 2],digits=2)
            profit_margins[i] = round(prices[t, j] - round(mc[t, j], digits=2), digits=2)
            i += 1
        end
    end

    results_table = DataFrame(
        "Market" => all_markets,
        "Product" => all_products,
        "Price" => all_prices,
        "Marginal Cost" => all_mc,
        "Market Share" => all_ms,
        "Characteristic 1" => all_char1,
        "Characteristic 2" => all_char2,
        "Profit Margin" => profit_margins
    )

    function write_table(io::IO, table::DataFrame, title::String)
        write(io, "\\begin{table}[h]\n")
        write(io, "\\centering\n")
        write(io, "\\caption{" * title * "}\n")
        write(io, "\\resizebox{1\\textwidth}{!}{")
        pretty_table(io, table; backend=Val(:latex))
        write(io, "}\\end{table}\n\n")
    end

    open(output_file, "w") do io
        write_table(io, results_table, "Task 1")
    end
    println("\nTables have been saved to '$output_file'")
end

main()
