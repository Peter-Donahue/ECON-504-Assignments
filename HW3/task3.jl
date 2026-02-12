import Pkg; Pkg.add(["Distributed", "Distributions", "Random", "JuMP", "Ipopt", "LinearAlgebra", "FastGaussQuadrature", "SparseGrids", "Printf", "PrettyTables", "DataFrames"])
_BASE_PATH = @__DIR__

using Distributed

CPU_cores_available = Sys.CPU_THREADS
#CPU_cores_to_use::Int = CPU_cores_available - 2
CPU_cores_to_use::Int = 20
println("Number of CPU cores on system: ",CPU_cores_available )
println("Number of CPU cores I'll use: ",CPU_cores_to_use)
addprocs(CPU_cores_to_use)

@everywhere begin
    using Random
    using LinearAlgebra
    using Statistics
    using Distributions
    using JuMP
    using Ipopt
    using FastGaussQuadrature
    using SparseGrids

    const _BETA_MEANS = [1.5, 1.0, 1.0, 1.0]
    const _BETA_SDS = [1.0, 1.0, 1.0, 0.2]

    const _SIGMA_XW = [0.25 0.05 0.05 0.05;
                    0.05 0.25 0.05 0.05;
                    0.05 0.05 0.1  0.05;
                    0.05 0.05 0.05 0.1 ]
    const _MEAN_XW = zeros(4)
    
    const JT_combinations = [
        (J = 5, T = 5),
        (J = 10, T = 10),
        (J = 15, T = 15),
        (J = 20, T = 20)
    ]

    #const _S = 10
    #const _R = 10
    #const _NUM_STARTING_VALUES = 2
    const _S = 1_000_000
    const _R = 100 
    const _NUM_STARTING_VALUES = 10
    
end

@everywhere begin

    function smolyak_sparse_grid(level, dimension)
        nodes, weights = sparsegrid(dimension, level, FastGaussQuadrature.gausshermite)
        return hcat(nodes...)', weights
    end

    function generate_dataset(J, T, S)
        x = zeros(T, J, 2)          # Product characteristics
        w = zeros(T, J, 2)          # Cost shifters
        mc = zeros(T, J)            # Marginal costs
        prices = zeros(T, J)        # Prices
        market_shares = zeros(T, J) # Market shares

        for t in 1:T
            xw = rand(MvNormal(_MEAN_XW, _SIGMA_XW), J)'

            x[t, :, :] = xw[:, 1:2] 
            w[t, :, :] = xw[:, 3:4]  

            log_mc = 0.5 .+ 0.1 .* w[t, :, 1] .+ 0.1 .* w[t, :, 2]
            mc[t, :] = exp.(log_mc)

            prices_t = mc[t, :]
            prices[t, :] = prices_t

            beta_i_t = zeros(S, 4)
            for s in 1:S
                beta_i_t[s, :] = [rand(Normal(_BETA_MEANS[k], _BETA_SDS[k])) for k in 1:4]
            end
            beta0_s = beta_i_t[:, 1]
            beta1_s = beta_i_t[:, 2]
            beta2_s = beta_i_t[:, 3]
            beta3_s = abs.(beta_i_t[:, 4])

            V = zeros(S, J)
            for s in 1:S
                for j in 1:J
                    V[s, j] = beta0_s[s] + beta1_s[s] * x[t, j, 1] + beta2_s[s] * x[t, j, 2] - beta3_s[s] * prices_t[j]
                end
            end
            exp_V = exp.(V)
            denom = 1 .+ sum(exp_V, dims=2)
            P = exp_V ./ denom
            market_shares[t, :] = mean(P, dims=1)
        end

        return (x = x, w = w, mc = mc, prices = prices, market_shares = market_shares)
    end

    function compute_instruments(x, w, J, T)
        num_instruments = 10 

        Z = zeros(T * J, num_instruments)

        x1 = reshape(x[:, :, 1], T, J)
        x2 = reshape(x[:, :, 2], T, J)
        w1 = reshape(w[:, :, 1], T, J)
        w2 = reshape(w[:, :, 2], T, J)
    
        for t in 1:T
            for j in 1:J
                idx = (t - 1) * J + j
    
                # (a) Intercept
                Z[idx, 1] = 1
    
                # (b) Marginal cost shifters for own product
                Z[idx, 2] = w1[t, j]
                Z[idx, 3] = w2[t, j]
    
                # (c) Squares of the marginal cost shifters
                Z[idx, 4] = w1[t, j]^2
                Z[idx, 5] = w2[t, j]^2
    
                # Indices of other products
                other_indices = [k for k in 1:J if k != j]
    
                # (d) Sum of characteristic differences
                sum_d_x1 = sum(x1[t, j] .- x1[t, other_indices])
                sum_d_x2 = sum(x2[t, j] .- x2[t, other_indices])
                Z[idx, 6] = sum_d_x1
                Z[idx, 7] = sum_d_x2
    
                # (e) Sum of squared characteristic differences
                sum_d_x1_sq = sum((x1[t, j] .- x1[t, other_indices]).^2)
                sum_d_x2_sq = sum((x2[t, j] .- x2[t, other_indices]).^2)
                Z[idx, 8] = sum_d_x1_sq
                Z[idx, 9] = sum_d_x2_sq
    
                # (f) Interaction term
                interaction_term = sum((x1[t, j] .- x1[t, other_indices]) .* (x2[t, j] .- x2[t, other_indices]))
                Z[idx, 10] = interaction_term
            end
        end
    
        return Z
    end

    function estimate_model(data, instruments, J, T, starting_values_list)

        x = data[:x]
        prices = data[:prices]
        market_shares = data[:market_shares]

        x1 = reshape(x[:, :, 1], T * J)
        x2 = reshape(x[:, :, 2], T * J)
        prices_vec = reshape(prices, T * J)
        shares_vec = reshape(market_shares, T * J)
    
        Z = instruments
    
        num_params = 8
    
        best_obj_value = Inf
        best_estimates = nothing
    
        for beta_init in starting_values_list
            model = Model(Ipopt.Optimizer)
            set_silent(model)
    
            @variable(model, beta_means[i=1:4], start=beta_init[i])
            @variable(model, sigma[i=1:4] >= 0, start=abs(beta_init[4 + i]))
            N_data = T * J
            nodes, weights = smolyak_sparse_grid(5, 4)
            N_nodes = size(nodes, 1)
            total_weight = sum(weights)

            @NLexpression(model, V[i=1:N_data, n=1:N_nodes],
                (beta_means[1] + sqrt(2)*sigma[1]*nodes[n,1])
                + (beta_means[2] + sqrt(2)*sigma[2]*nodes[n,2]) * x1[i]
                + (beta_means[3] + sqrt(2)*sigma[3]*nodes[n,3]) * x2[i]
                - abs(beta_means[4] + sqrt(2)*sigma[4]*nodes[n,4]) * prices_vec[i]
            )
    
            @NLexpression(model, exp_V[i=1:N_data, n=1:N_nodes], exp(V[i, n]))
            @NLexpression(model, denom[i=1:N_data, n=1:N_nodes],
                1 + exp_V[i, n]
            )
            @NLexpression(model, P[i=1:N_data, n=1:N_nodes],
                exp_V[i, n] / denom[i, n]
            )
    
            @NLexpression(model, s_hat[i=1:N_data],
                sum(weights[n] * P[i, n] for n = 1:N_nodes) / total_weight
            )
    
            num_instruments = size(Z, 2)
            @NLexpression(model, moments[i=1:N_data, j=1:num_instruments],
                (shares_vec[i] - s_hat[i]) * Z[i, j]
            )
    
            W = Matrix{Float64}(I, num_instruments, num_instruments)

            @NLobjective(model, Min, sum(
                moments[i, j] * W[j, k] * moments[i, k] for i in 1:N_data, j in 1:num_instruments, k in 1:num_instruments
            ))
    
            optimize!(model)
            termination_status = JuMP.termination_status(model)
            if termination_status == MOI.OPTIMAL || termination_status == MOI.LOCALLY_SOLVED
                obj_value = objective_value(model)
                if obj_value < best_obj_value
                    best_obj_value = obj_value
                    beta_means_est = value.(beta_means)
                    sigma_est = value.(sigma)
                    s_hat_values = [value(s_hat[i]) for i in 1:N_data]  # Extract s_hat values
                    best_estimates = (beta_means_est = beta_means_est, sigma_est = sigma_est, s_hat = s_hat_values)
                    println("Solver found an optimal solution\n","starting_value=", beta_init, "\nobj_value = ", best_estimates, "\nbest_estimate = ", best_estimates)
                end
            else
                println("Solver did not find an optimal solution with starting values: ", beta_init)
            end
        end
    
        if best_estimates === nothing
            error("Failed to find a feasible solution with any starting values.")
        end

        residuals = shares_vec .- best_estimates.s_hat  
        moments_matrix = residuals .* Z  
        W_opt = inv((moments_matrix' * moments_matrix) / (T * J))
        model = Model(Ipopt.Optimizer)
        set_silent(model)
    
        @variable(model, beta_means[i=1:4], start=best_estimates.beta_means_est[i])
        @variable(model, sigma[i=1:4] >= 0, start=best_estimates.sigma_est[i])

        @NLexpression(model, V[i=1:N_data, n=1:N_nodes],
            (beta_means[1] + sqrt(2)*sigma[1]*nodes[n,1])
            + (beta_means[2] + sqrt(2)*sigma[2]*nodes[n,2]) * x1[i]
            + (beta_means[3] + sqrt(2)*sigma[3]*nodes[n,3]) * x2[i]
            - abs(beta_means[4] + sqrt(2)*sigma[4]*nodes[n,4]) * prices_vec[i]
        )
    
        @NLexpression(model, exp_V[i=1:N_data, n=1:N_nodes], exp(V[i, n]))
        @NLexpression(model, denom[i=1:N_data, n=1:N_nodes],
            1 + exp_V[i, n]
        )
        @NLexpression(model, P[i=1:N_data, n=1:N_nodes],
            exp_V[i, n] / denom[i, n]
        )
    
        @NLexpression(model, s_hat[i=1:N_data],
            sum(weights[n] * P[i, n] for n = 1:N_nodes) / total_weight
        )
 
        @NLexpression(model, moments[i=1:N_data, j=1:num_instruments],
            (shares_vec[i] - s_hat[i]) * Z[i, j]
        )

        @NLobjective(model, Min, sum(
            moments[i, j] * W_opt[j, k] * moments[i, k] for i in 1:N_data, j in 1:num_instruments, k in 1:num_instruments
        ))

        optimize!(model)
    
        termination_status = JuMP.termination_status(model)
        if termination_status == MOI.OPTIMAL || termination_status == MOI.LOCALLY_SOLVED
            beta_means_est = value.(beta_means)
            sigma_est = value.(sigma)
            s_hat_values = [value(s_hat[i]) for i in 1:N_data] 
            final_estimates = (beta_means_est = beta_means_est, sigma_est = sigma_est, s_hat = s_hat_values)
        else
            error("Second-stage estimation failed to find an optimal solution.")
        end
    
        return final_estimates
    end
    
    function process_single_simulation(J, T, sim_number, S)
        Random.seed!(1234 + sim_number)
        data = generate_dataset(J, T, S)
        instruments = compute_instruments(data[:x], data[:w], J, T)
        starting_values_list = []
        for i in 1:_NUM_STARTING_VALUES
            beta_init = rand(Uniform(-1, 1), 8)
            push!(starting_values_list, beta_init)
        end
        estimates = estimate_model(data, instruments, J, T, starting_values_list)
        return estimates
    end
end  # End of @everywhere block

function main()
    @sync begin  
        for (J_T) in JT_combinations
            @async begin
                J = J_T.J
                T = J_T.T

                println("Processing combination J=$J, T=$T...")

                simulations = 1:_R

                results = pmap(sim -> process_single_simulation(J, T, sim, _S), simulations)

                beta_means_estimates = [result.beta_means_est for result in results]
                sigma_estimates = [result.sigma_est for result in results]

                beta_means_array = hcat(beta_means_estimates...)
                sigma_array = hcat(sigma_estimates...)

                beta_means_mean = mean(beta_means_array, dims=2)
                beta_means_std = std(beta_means_array, dims=2)

                sigma_mean = mean(sigma_array, dims=2)
                sigma_std = std(sigma_array, dims=2)

                println("Results for J=$J, T=$T:")
                println("Means of beta_means: ", beta_means_mean)
                println("Standard deviations of beta_means: ", beta_means_std)
                println("Means of sigma: ", sigma_mean)
                println("Standard deviations of sigma: ", sigma_std)
            end
        end
    end
end

function counterfactual(J, T, S)
    println("Starting Merger Counterfactual Analysis...")

    data = generate_dataset(J, T, S)

    instruments = compute_instruments(data[:x], data[:w], J, T)

    starting_values_list = []
    for i in 1:_NUM_STARTING_VALUES
        beta_init = rand(Uniform(-1, 1), 8)
        push!(starting_values_list, beta_init)
    end

    final_estimates = estimate_model(data, instruments, J, T, starting_values_list)
    pre_merger_prices = compute_equilibrium_prices(final_estimates, data, J, T)
    println("Computed pre-merger prices.")
    post_merger_prices = compute_post_merger_prices(final_estimates, data, J, T)
    println("Computed post-merger prices.")
    pre_merger_shares = compute_market_shares(pre_merger_prices, final_estimates, data, J, T)
    post_merger_shares = compute_market_shares(post_merger_prices, final_estimates, data, J, T)
    println("Computed market shares.")

    x = data[:x]
    for j in 3:J
        x1_j = mean(x[:, j, 1])
        x2_j = mean(x[:, j, 2])

        x1_merge = mean(mean(x[:, 1:2, 1], dims=2))
        x2_merge = mean(mean(x[:, 1:2, 2], dims=2))

        diff_x1 = x1_j - x1_merge
        diff_x2 = x2_j - x2_merge

        avg_price_change = mean(post_merger_prices[:, j] - pre_merger_prices[:, j])
        avg_share_change = mean(post_merger_shares[:, j] - pre_merger_shares[:, j])

        println("Non-merging good $j:")
        println("  Difference in characteristic 1: ", diff_x1)
        println("  Difference in characteristic 2: ", diff_x2)
        println("  Average price change: ", avg_price_change)
        println("  Average market share change: ", avg_share_change)
        println()
    end

    for j in 1:2
        avg_price_change = mean(post_merger_prices[:, j] - pre_merger_prices[:, j])
        avg_share_change = mean(post_merger_shares[:, j] - pre_merger_shares[:, j])

        println("Merged good $j:")
        println("  Average price change: ", avg_price_change)
        println("  Average market share change: ", avg_share_change)
        println()
    end
end

@everywhere begin
    const NODES, WEIGHTS = smolyak_sparse_grid(5, 4)
    const N_nodes = size(NODES, 1)
    const total_weight = sum(WEIGHTS)
end
# Run the main function
main() 
counterfactual(10, 25, 10)