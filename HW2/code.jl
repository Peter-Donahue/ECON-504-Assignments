########### Import Statements ######################
#import Pkg; Pkg.add(["CSV", "Memoize", "Distributions", "Random", "LinearAlgebra", "JuMP", "HiGHS", "BlackBoxOptim", "DataFrames", "StatsBase", "Statistics", "SharedArrays", "Dates", "BenchmarkTools", "PrettyTables"])
###########

using Distributed
using Profile
using PrettyTables
using Distributions

CPU_cores_available = Sys.CPU_THREADS
CPU_cores_to_use::Int = CPU_cores_available ÷ 2
#CPU_cores_to_use::Int = 15
println("Number of CPU cores on system: ",CPU_cores_available )
println("Number of CPU cores I'll use: ",CPU_cores_to_use)
addprocs(CPU_cores_to_use)

_ROOT_DIR = @__DIR__

@everywhere begin
    using Random 
    using LinearAlgebra
    using JuMP
    using HiGHS
    using DataFrames
    using CSV
    using StatsBase
    using Statistics
    using SharedArrays
    import Dates 
    #using Optim
    using BlackBoxOptim
    using Memoize

    # Other Constants used while debugging
    using BenchmarkTools
    const _SILENCE_MODEL = true
    const _DEBUG_VERBOSE = 0
    const _REGENERATE_REPLICATION_DATA = false

    _ROOT_DIR = $_ROOT_DIR
    const _DATA_FOLDER_NAME = "data"
    const _REPLICATION_DATA_FOLDER = "replications"
    const _FAKE_DATASET_FILE_NAME ="fake_matching_data.csv"
    const _MOMENTS_DATASET_FILE_NAME = "moments_data.csv"
    const _RANDOM_GEN_SEED_BASE = 1234


    
    #generate_market_data_runs::Int64 = 0
    # Set parameters
    const _N = 10                        # Number of agents on each side
    const _T = 100                      # Number of markets
    const _S = 500                     # Number of simulations
    const _R = 100                       #Number of Monte-Carlo replications
    const _GAMMA = [1.0, -2.0, 2.0, 1.5] # Coefficient vector
    const _RHO_1 = 0.2                    # Correlation parameter
    const _RHO_2 = 0.4
    const _RHO_3 = 0.6
    const _SIGMA = 2.0                   # Standard deviation of unobserved complementarities
    #const _THETA_0 = [_RHO_1, _RHO_2, _RHO_3, _SIGMA, _GAMMA]
    const _THETA_0 = [_RHO_1, _RHO_2, _RHO_3, _SIGMA, _GAMMA[1], _GAMMA[2], _GAMMA[3], _GAMMA[4]]

    const _SEARCH_RANGE_LB = [0.0, 0.0, 0.0, 0.0, -5.0, -5.0, -5.0, -5.0]
    const _SEARCH_RANGE_UB = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    const _NUM_MOMENTS = 87
    const _MOMENT_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]

    const _BBOPTIMIZE_MAX_STEPS = 25
    const _BBOPTIMIZE_MAX_TIME = 10000
    const _BBOPTIMIZE_POPULATION_SIZE = 50
end


@everywhere begin
    function generate_covariance(N::Int64, sigma::Float64, rho1::Float64, 
            rho2::Float64, rho3::Float64)

        num_elements = (N - 1) * (N - 1)
        sigma_B = zeros(num_elements, num_elements)
        indices = [(u, d) for u in 2:N, d in 2:N]
        
        for i in 1:num_elements 
            
            (u1, d1) = indices[i]

            for j in i:num_elements
                (u2, d2) = indices[j]
                if u1 == u2 && d1 == d2
                    cov = sigma^2
                elseif u1 == u2 && d1 != d2
                    cov = sigma^2 * rho3
                elseif u1 != u2 && d1 == d2
                    cov = sigma^2 * rho2
                else  # u1 != u2 && d1 != d2
                    cov = sigma^2 * rho1
                end
                # Symmetric assignment
                sigma_B[i, j] = cov
                sigma_B[j, i] = cov
            end
            
        end
        return sigma_B
    end

    #@memoize function generate_market_data(N::Int64, K::Int64, seed_id::Int64)
    function generate_market_data(N::Int64, K::Int64)
        #generate_market_data_runs += 1
        Z_u = randn(N, 2) 
        Z_d = randn(N, 2)
        Z_match = randn(N, N, 2)

        Z_ij = Array{Float64, 3}(undef, N, N, K)
        for i in 1:N
            for j in 1:N
                Z_ij[i, j, 1] = Z_match[i, j, 1]
                Z_ij[i, j, 2] = Z_match[i, j, 2]
                Z_ij[i, j, 3] = Z_u[i, 1] * Z_d[j, 1]
                Z_ij[i, j, 4] = Z_u[i, 2] * Z_d[j, 2]
            end
        end
        return Z_u, Z_d, Z_ij
    end

    function create_matching_model(N::Int64)
        model = Model(HiGHS.Optimizer)
        if _SILENCE_MODEL
            set_silent(model)
        end
        @variable(model, H[1:N, 1:N], Bin)
        @constraint(model, [i=1:N], sum(H[i,j] for j in 1:N) == 1)
        @constraint(model, [j=1:N], sum(H[i,j] for i in 1:N) == 1)
        @constraint(model, [i=1:N, j=1:N], H[i,j] >= 0)
        return model, H
    end

    function solve_matching(N::Int64, gamma::Vector{Float64}, 
            Z_ij::Array{Float64, 3}, B::Array{Float64, 2}, 
            model::Model, H::Array{VariableRef, 2})

        W_ij = sum(Z_ij .* reshape(gamma, (1, 1, length(gamma))); dims=3)
        delta = W_ij[1,1] .+ W_ij .- W_ij[1, :] .- W_ij[:, 1]
        b = [zeros(1, N); zeros(N - 1, 1) B]
        b_w = delta + b
        @objective(model, Min, sum(b_w .* H))
        set_objective_sense(model, MOI.MAX_SENSE)
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            error("Optimization did not find an optimal solution.")
        end
        matches = Tuple.(findall(value.(H) .> 0.5))
        return matches
    end

    function generate_market_replication(r::Int64, t::Int64, N::Int64, K::Int64, 
            gamma::Vector{Float64}, chol_sigma_B::Cholesky{Float64, Matrix{Float64}},
            model::Model, H::Matrix{VariableRef})

        if _DEBUG_VERBOSE == 1
            println("replication $r, market $t")
        end
        seed_id = r*100 + t
        Random.seed!(seed_id)

        # Generate B_{s,t}
        num_elements = (N - 1) * (N - 1)
        z_b = randn(num_elements)
        vec_B = chol_sigma_B.L * z_b
        B = reshape(vec_B, N - 1, N - 1)

        #Z_u, Z_d, Z_ij = generate_market_data(N, K, seed_id)
        Z_u, Z_d, Z_ij = generate_market_data(N, K)
        #model, H = create_matching_model(N)

        matches = solve_matching(N, gamma, Z_ij, B, model, H)

        total_rows = N * N
        data = DataFrame(
            replication_id = Vector{Int}(undef, total_rows),
            market_id = Vector{Int}(undef, total_rows),
            upstream_id = Vector{Int}(undef, total_rows),
            downstream_id = Vector{Int}(undef, total_rows),
            Z_u1 = Vector{Float64}(undef, total_rows),
            Z_u2 = Vector{Float64}(undef, total_rows),
            Z_d1 = Vector{Float64}(undef, total_rows),
            Z_d2 = Vector{Float64}(undef, total_rows),
            Z_match1 = Vector{Float64}(undef, total_rows),
            Z_match2 = Vector{Float64}(undef, total_rows),
            W_ij = Vector{Float64}(undef, total_rows),
            matched = Vector{Int}(undef, total_rows), 
        )

        matched_pairs = Set{Tuple{Int, Int}}(matches)
        row_index = 1

        for i in 1:N
            for j in 1:N
                Z_ui = Z_u[i, :]
                Z_dj = Z_d[j, :]
                Z_ij_match = Z_ij[i, j, :]

                W_obs = dot(Z_ij_match, gamma)

                is_matched = (i, j) in matched_pairs ? 1 : 0

                data.replication_id[row_index] = r 
                data.market_id[row_index] = t
                data.upstream_id[row_index] = i
                data.downstream_id[row_index] = j
                data.Z_u1[row_index] = Z_ui[1]
                data.Z_u2[row_index] = Z_ui[2]
                data.Z_d1[row_index] = Z_dj[1]
                data.Z_d2[row_index] = Z_dj[2]
                data.Z_match1[row_index] = Z_ij_match[1]
                data.Z_match2[row_index] = Z_ij_match[2]
                data.W_ij[row_index] = W_obs
                data.matched[row_index] = is_matched

                row_index += 1
            end
        end
        return data
    end

    function generate_market_simluation(r::Int64, s::Int64, t::Int64, N::Int64, K::Int64, 
        gamma::Vector{Float64}, chol_sigma_B::Cholesky{Float64, Matrix{Float64}},
        model::Model, H::Matrix{VariableRef})

        if _DEBUG_VERBOSE == 1
            println("replication $r, simulation $s, market $t")
        end
        seed_id = r*1000000 + s*1000 + t
        Random.seed!(r*1000000 + s*1000 + t)

        # Generate B_{s,t}
        num_elements = (N - 1) * (N - 1)
        z_b = randn(num_elements)
        vec_B = chol_sigma_B.L * z_b
        B = reshape(vec_B, N - 1, N - 1)

        #Z_u, Z_d, Z_ij = generate_market_data(N, K, seed_id)
        Z_u, Z_d, Z_ij = generate_market_data(N, K)
        #model, H = create_matching_model(N)
        
        matches = solve_matching(N, gamma, Z_ij, B, model, H)

        total_rows = N * N
        data = DataFrame(
            replication_id = Vector{Int}(undef, total_rows),
            simulation_id = Vector{Int}(undef, total_rows),
            market_id = Vector{Int}(undef, total_rows),
            upstream_id = Vector{Int}(undef, total_rows),
            downstream_id = Vector{Int}(undef, total_rows),
            Z_u1 = Vector{Float64}(undef, total_rows),
            Z_u2 = Vector{Float64}(undef, total_rows),
            Z_d1 = Vector{Float64}(undef, total_rows),
            Z_d2 = Vector{Float64}(undef, total_rows),
            Z_match1 = Vector{Float64}(undef, total_rows),
            Z_match2 = Vector{Float64}(undef, total_rows),
            W_ij = Vector{Float64}(undef, total_rows),
            matched = Vector{Int}(undef, total_rows), 
        )

        matched_pairs = Set{Tuple{Int, Int}}(matches)
        row_index = 1

        for i in 1:N
            for j in 1:N
                Z_ui = Z_u[i, :]
                Z_dj = Z_d[j, :]
                Z_ij_match = Z_ij[i, j, :]

                W_obs = dot(Z_ij_match, gamma)

                is_matched = (i, j) in matched_pairs ? 1 : 0

                data.replication_id[row_index] = r 
                data.simulation_id[row_index] = s 
                data.market_id[row_index] = t
                data.upstream_id[row_index] = i
                data.downstream_id[row_index] = j
                data.Z_u1[row_index] = Z_ui[1]
                data.Z_u2[row_index] = Z_ui[2]
                data.Z_d1[row_index] = Z_dj[1]
                data.Z_d2[row_index] = Z_dj[2]
                data.Z_match1[row_index] = Z_ij_match[1]
                data.Z_match2[row_index] = Z_ij_match[2]
                data.W_ij[row_index] = W_obs
                data.matched[row_index] = is_matched

                row_index += 1
            end
        end
        return data
    end

    function compute_market_moments(market_data::DataFrame)
        moment_quantiles::Vector{Float64} = _MOMENT_QUANTILES
        N = length(unique(market_data.upstream_id))
        match_characters = [:Z_match1, :Z_match2]
        agent_characters = [:Z_u1, :Z_u2]
        moments = zeros(_NUM_MOMENTS)
        moment_index = 1
    
        observed_matches = market_data[market_data.matched .== 1, :]
    
        L_match = Dict{Symbol, Vector{Float64}}()
        L_agent = Dict{Symbol, Vector{Float64}}()
        L_agent_up = Dict{Symbol, Vector{Float64}}()
        L_agent_down = Dict{Symbol, Vector{Float64}}()
    
        for k in match_characters
            L_match[k] = observed_matches[!, k]
        end
        for k in agent_characters
            L_agent[k] = observed_matches[!, k] .* observed_matches[!, Symbol(replace(string(k), "Z_u" => "Z_d"))]
            L_agent_up[k] = observed_matches[!, k]
            L_agent_down[k] = observed_matches[!, Symbol(replace(string(k), "Z_u" => "Z_d"))]
        end

        ###
        #function_times = Dict{String, Float64}()

        ## Quantile Moments for Match-specific Characteristics 
        ## Moments 1-10
        #function_times["compute_quantile_moments_1"] = @elapsed moment_index = compute_quantile_moments!(L_match, moment_quantiles, moments, moment_index, match_characters)
        moment_index = compute_quantile_moments!(L_match, moment_quantiles, moments, moment_index, match_characters)

        ## Quantile Moments for Agent-specific Characteristics
        ## Moments 11-20
        #function_times["compute_quantile_moments_2"] = @elapsed moment_index = compute_quantile_moments!(L_agent, moment_quantiles, moments, moment_index, agent_characters)
        moment_index = compute_quantile_moments!(L_agent, moment_quantiles, moments, moment_index, agent_characters)

        ## Correlation Moments 
        ## Moments 21-27
        #function_times["compute_regression_moments"] = @elapsed moment_index = compute_correlation_moments!(L_match, L_agent_up, L_agent_down, L_agent, moments, moment_index, match_characters, agent_characters)
        moment_index = compute_correlation_moments!(L_match, L_agent_up, L_agent_down, L_agent, moments, moment_index, match_characters, agent_characters)

        ## Regression Moments
        ## Moments 28-31
        #function_times["compute_regression_moments"] = @elapsed moment_index = compute_regression_moments!(market_data, moments, moment_index, match_characters, agent_characters)
        moment_index = compute_regression_moments!(market_data, moments, moment_index, match_characters, agent_characters)

        ## Opportunity Cost Moments for Match-specific Characteristics
        ## Moments 32-51
        #function_times["compute_opportunity_cost_moments_match"] = @elapsed moment_index = compute_opportunity_cost_moments_match!(market_data, observed_matches, N, moments, moment_index, match_characters, moment_quantiles)
        moment_index = compute_opportunity_cost_moments_match!(market_data, observed_matches, N, moments, moment_index, match_characters, moment_quantiles)

        ## Opportunity Cost Moments for Agent-specific Characteristics
        ## Moments 52-71
        #function_times["compute_opportunity_cost_moments_agent"] = @elapsed moment_index = compute_opportunity_cost_moments_agent!(market_data, observed_matches, N, moments, moment_index, agent_characters, moment_quantiles)
        moment_index = compute_opportunity_cost_moments_agent!(market_data, observed_matches, N, moments, moment_index, agent_characters, moment_quantiles)

        ## Regression Moments - Opportunity Cost
        ## Moments 72-75
        #function_times["compute_regression_moments_opportunity_cost"] = @elapsed moment_index = compute_regression_moments_opportunity_cost!(market_data, moments, moment_index, match_characters, agent_characters)
        moment_index = compute_regression_moments_opportunity_cost!(market_data, moments, moment_index, match_characters, agent_characters)

        ## Rank Moments 
        ## Moments 76-87
        #function_times["compute_rank_moments"] = @elapsed moment_index = compute_rank_moments!(market_data, observed_matches, N, moments, moment_index, match_characters, agent_characters)
        moment_index = compute_rank_moments!(market_data, observed_matches, N, moments, moment_index, match_characters, agent_characters)

        # slowest::Tuple{String, Float64} = ("none", 0)
        # for (i,kv) in enumerate(function_times)
        #     if kv[2] > slowest[2]
        #         slowest = kv[1], kv[2]
        #     end
        # end
        # println("slowest function = $(slowest[1]) ($(slowest[2]))")

        return moments
    end

    function compute_quantile_moments!(L_values::Dict{Symbol, Vector{Float64}}, 
            moment_quantiles::Vector{Float64}, moments::Vector{Float64}, 
            moment_index::Int64, characteristics::Vector{Symbol})

        for k in characteristics
            for p in moment_quantiles
                moments[moment_index] = quantile(L_values[k], p)
                moment_index += 1
            end
        end
        return moment_index
    end
    
    function compute_correlation_moments!(L_match::Dict{Symbol, Vector{Float64}}, 
            L_agent_up::Dict{Symbol, Vector{Float64}}, L_agent_down::Dict{Symbol, Vector{Float64}}, 
            L_agent::Dict{Symbol, Vector{Float64}}, moments::Vector{Float64}, 
            moment_index::Int64, match_characters::Vector{Symbol}, agent_characters::Vector{Symbol})

        for i in eachindex(match_characters)
            for j in eachindex(match_characters)
                if j > i
                    k1 = match_characters[i]
                    k2 = match_characters[j]
                    moments[moment_index] = cor(L_match[k1], L_match[k2])
                    moment_index += 1
                end
            end
        end

        for k in agent_characters
            moments[moment_index] = cor(L_agent_up[k], L_agent_down[k])
            moment_index += 1
        end

        for k_match in match_characters
            for k_agent in agent_characters
                moments[moment_index] = cor(L_match[k_match], L_agent[k_agent])
                moment_index += 1
            end
        end
        return moment_index
    end
    
    function compute_regression_moments!(market_data::DataFrame, moments::Vector{Float64}, 
            moment_index::Int64, match_characters::Vector{Symbol}, agent_characters::Vector{Symbol})

        matches_indicator = market_data.matched
    
        regression_df = DataFrame(matches_indicator = matches_indicator)
    
        for k in match_characters
            regression_df[!, string(k)] = market_data[!, k]
        end
        for k in agent_characters
            interaction_term = market_data[!, k] .* market_data[!, Symbol(replace(string(k), "Z_u" => "Z_d"))]
            regression_df[!, string(k) * "_interaction"] = interaction_term
        end
    
        if all(matches_indicator .== 1) || all(matches_indicator .== 0)
            coefficients = zeros(4)
        else
            X = hcat(
                regression_df[!, string(match_characters[1])],
                regression_df[!, string(match_characters[2])],
                regression_df[!, string(agent_characters[1]) * "_interaction"],
                regression_df[!, string(agent_characters[2]) * "_interaction"]
            )
            y = regression_df.matches_indicator
            X = hcat(ones(size(X, 1)), X)
            beta = inv(X' * X) * (X' * y)
            coefficients = beta[2:end] 
        end

        for coef in coefficients
            moments[moment_index] = coef
            moment_index += 1
        end
        return moment_index
    end
    
    function compute_opportunity_cost_moments_match!(market_data::DataFrame, 
            observed_matches::DataFrame, N::Int64, moments::Vector{Float64}, 
            moment_index::Int64, match_characters::Vector{Symbol}, 
            moment_quantiles::Vector{Float64})

        L_match_opp_up = Dict{Symbol, Vector{Float64}}()
        L_match_opp_down = Dict{Symbol, Vector{Float64}}()
    
        for k in match_characters
            L_match_opp_up_k = zeros(N)
            L_match_opp_down_k = zeros(N)
            for i in 1:N
                matched_j = observed_matches.downstream_id[observed_matches.upstream_id .== i][1]
                opp_values_up = market_data[(market_data.upstream_id .== i) .&& (market_data.downstream_id .!= matched_j), k]
                if isempty(opp_values_up)
                    L_match_opp_up_k[i] = NaN
                else
                    L_match_opp_up_k[i] = mean(opp_values_up)
                end

                matched_i = observed_matches.upstream_id[observed_matches.downstream_id .== matched_j][1]
                opp_values_down = market_data[(market_data.downstream_id .== matched_j) .&& (market_data.upstream_id .!= matched_i), k]
                if isempty(opp_values_down)
                    L_match_opp_down_k[i] = NaN
                else
                    L_match_opp_down_k[i] = mean(opp_values_down)
                end
            end

            L_match_opp_up[k] = L_match_opp_up_k
            L_match_opp_down[k] = L_match_opp_down_k
        end
    
        for k in match_characters
            for p in moment_quantiles
    
                valid_values_up = filter(!isnan, L_match_opp_up[k])
                moments[moment_index] = isempty(valid_values_up) ? NaN : quantile(valid_values_up, p)
                moment_index += 1
    
                valid_values_down = filter(!isnan, L_match_opp_down[k])
                moments[moment_index] = isempty(valid_values_down) ? NaN : quantile(valid_values_down, p)
                moment_index += 1
            end
        end
    
        return moment_index
    end
    
    function compute_opportunity_cost_moments_agent!(market_data::DataFrame, 
            observed_matches::DataFrame, N::Int64, moments::Vector{Float64}, 
            moment_index::Int64, agent_characters::Vector{Symbol}, moment_quantiles::Vector{Float64})

        L_agent_opp_up = Dict{Symbol, Vector{Float64}}()
        L_agent_opp_down = Dict{Symbol, Vector{Float64}}()
    
        for k in agent_characters
            L_agent_opp_up_k = zeros(N)
            L_agent_opp_down_k = zeros(N)
            for i in 1:N
                matched_j = observed_matches.downstream_id[observed_matches.upstream_id .== i][1]
                Z_i_k = observed_matches[observed_matches.upstream_id .== i, k][1]
                opp_values_up = market_data[(market_data.upstream_id .== i) .&& (market_data.downstream_id .!= matched_j), Symbol(replace(string(k), "Z_u" => "Z_d"))]
                if isempty(opp_values_up)
                    L_agent_opp_up_k[i] = NaN
                else
                    interaction_terms = Z_i_k .* opp_values_up
                    L_agent_opp_up_k[i] = mean(interaction_terms)
                end
    
                matched_i = observed_matches.upstream_id[observed_matches.downstream_id .== matched_j][1]
                Z_j_k = observed_matches[observed_matches.downstream_id .== matched_j, Symbol(replace(string(k), "Z_u" => "Z_d"))][1]
                opp_values_down = market_data[(market_data.downstream_id .== matched_j) .&& (market_data.upstream_id .!= matched_i), k]
                if isempty(opp_values_down)
                    L_agent_opp_down_k[i] = NaN
                else
                    interaction_terms = opp_values_down .* Z_j_k
                    L_agent_opp_down_k[i] = mean(interaction_terms)
                end
            end
            L_agent_opp_up[k] = L_agent_opp_up_k
            L_agent_opp_down[k] = L_agent_opp_down_k
        end
    
        for k in agent_characters
            for p in moment_quantiles
                valid_values_up = filter(!isnan, L_agent_opp_up[k])
                moments[moment_index] = isempty(valid_values_up) ? NaN : quantile(valid_values_up, p)
                moment_index += 1

                valid_values_down = filter(!isnan, L_agent_opp_down[k])
                moments[moment_index] = isempty(valid_values_down) ? NaN : quantile(valid_values_down, p)
                moment_index += 1
            end
        end
    
        return moment_index
    end
             
    function compute_regression_moments_opportunity_cost!(market_data::DataFrame, 
            moments::Vector{Float64}, moment_index::Int64, match_characters::Vector{Symbol}, 
            agent_characters::Vector{Symbol})

        regression_df_opp = DataFrame(matches_indicator = market_data.matched)
        for k in match_characters
            opp_up = zeros(size(market_data, 1))
            for idx_row in 1:size(market_data, 1)
                i = market_data.upstream_id[idx_row]
                j = market_data.downstream_id[idx_row]
                opp_values = market_data[(market_data.upstream_id .== i) .&& (market_data.downstream_id .!= j), k]
                opp_up[idx_row] = isempty(opp_values) ? NaN : mean(opp_values)
            end
            regression_df_opp[!, "opp_up_" * string(k)] = opp_up
        
            opp_down = zeros(size(market_data, 1))
            for idx_row in 1:size(market_data, 1)
                i = market_data.upstream_id[idx_row]
                j = market_data.downstream_id[idx_row]
                opp_values = market_data[(market_data.downstream_id .== j) .&& (market_data.upstream_id .!= i), k]
                opp_down[idx_row] = isempty(opp_values) ? NaN : mean(opp_values)
            end
            regression_df_opp[!, "opp_down_" * string(k)] = opp_down
        end
        for k in agent_characters
        
            opp_agent_up = zeros(size(market_data, 1))
            for idx_row in 1:size(market_data, 1)
                i = market_data.upstream_id[idx_row]
                j = market_data.downstream_id[idx_row]
                Z_i_k = market_data[market_data.upstream_id .== i .&& market_data.downstream_id .== j, k][1]
                opp_values = market_data[(market_data.upstream_id .== i) .&& (market_data.downstream_id .!= j), Symbol(replace(string(k), "Z_u" => "Z_d"))]
                if isempty(opp_values)
                    opp_agent_up[idx_row] = NaN
                else
                    interaction_terms = Z_i_k .* opp_values
                    opp_agent_up[idx_row] = mean(interaction_terms)
                end
            end
            regression_df_opp[!, "opp_agent_up_" * string(k)] = opp_agent_up
        
            opp_agent_down = zeros(size(market_data, 1))
            for idx_row in 1:size(market_data, 1)
                i = market_data.upstream_id[idx_row]
                j = market_data.downstream_id[idx_row]
                Z_j_k = market_data[market_data.upstream_id .== i .&& market_data.downstream_id .== j, Symbol(replace(string(k), "Z_u" => "Z_d"))][1]
                opp_values = market_data[(market_data.downstream_id .== j) .&& (market_data.upstream_id .!= i), k]
                if isempty(opp_values)
                    opp_agent_down[idx_row] = NaN
                else
                    interaction_terms = opp_values .* Z_j_k
                    opp_agent_down[idx_row] = mean(interaction_terms)
                end
            end
            regression_df_opp[!, "opp_agent_down_" * string(k)] = opp_agent_down
        end

        complete_cases = dropmissing(regression_df_opp, disallowmissing=true)
    
        matches_indicator = complete_cases.matches_indicator
        if all(matches_indicator .== 1) || all(matches_indicator .== 0) || nrow(complete_cases) == 0
            coefficients_opp = zeros(4)
        else
            X_opp = hcat(
                complete_cases[!, "opp_up_" * string(match_characters[1])],
                complete_cases[!, "opp_down_" * string(match_characters[1])],
                complete_cases[!, "opp_agent_up_" * string(agent_characters[1])],
                complete_cases[!, "opp_agent_down_" * string(agent_characters[1])]
            )
            y_opp = complete_cases.matches_indicator
            X_opp = hcat(ones(size(X_opp, 1)), X_opp)
            beta_opp = inv(X_opp' * X_opp) * (X_opp' * y_opp)
            coefficients_opp = beta_opp[2:end]
        end
        for coef in coefficients_opp
            moments[moment_index] = coef
            moment_index += 1
        end
        return moment_index
    end
    
    function compute_rank_moments!(market_data::DataFrame, observed_matches::DataFrame, N::Int64, moments::Vector{Float64}, 
            moment_index::Int64, match_characters::Vector{Symbol}, agent_characters::Vector{Symbol})

        L_rank_up = Dict{Symbol, Vector{Float64}}()
        L_rank_down = Dict{Symbol, Vector{Float64}}()
    
        for k in match_characters
            ranks_up = zeros(N)
            for j in 1:N
                col_values = market_data[market_data.downstream_id .== j, k]
                ranks = tiedrank(col_values)
                i_indices = market_data[market_data.downstream_id .== j, :].upstream_id
                for idx_rank in eachindex(ranks)
                    i = i_indices[idx_rank]
                    ranks_up[i] = ranks[idx_rank]
                end
            end
            L_rank_up[k] = ranks_up

            ranks_down = zeros(N)
            for i in 1:N
                row_values = market_data[market_data.upstream_id .== i, k]
                ranks = tiedrank(row_values)
                j_indices = market_data[market_data.upstream_id .== i, :].downstream_id
                for idx_rank in eachindex(ranks)
                    j = j_indices[idx_rank]
                    ranks_down[j] = ranks[idx_rank]
                end
            end
            L_rank_down[k] = ranks_down
        end
  
        for k in match_characters
            moments[moment_index] = mean(L_rank_up[k]) / N
            moment_index += 1

            moments[moment_index] = var(L_rank_up[k]) / N
            moment_index += 1

            moments[moment_index] = mean(L_rank_down[k]) / N
            moment_index += 1

            moments[moment_index] = var(L_rank_down[k]) / N
            moment_index += 1
        end
    

        for k in agent_characters
            ranks_u = tiedrank(observed_matches[!, k])
            ranks_d = tiedrank(observed_matches[!, Symbol(replace(string(k), "Z_u" => "Z_d"))])
            rank_diff = abs.(ranks_u - ranks_d)

            moments[moment_index] = mean(rank_diff) / N
            moment_index += 1

            moments[moment_index] = var(rank_diff) / N
            moment_index += 1
        end
    
        return moment_index
    end
    
    function compute_all_markets_moments(data)
        
        T = length(unique(data.market_id))
        all_moments = SharedArray{Float64}(T, _NUM_MOMENTS)

        @sync @distributed for t in 1:T
            market_moments = compute_market_moments(data[data.market_id .== t, :])
            all_moments[t, :] .= market_moments
        end

        return [Vector(all_moments[i, :]) for i in 1:T]
    end

    function compute_all_simulation_moments_for_market(data)
        
        S = length(unique(data.simulation_id))
        all_moments = SharedArray{Float64}(S, _NUM_MOMENTS)

        @sync @distributed for s in 1:S
            market_moments = compute_market_moments(data[data.simulation_id .== s, :])
            all_moments[s, :] .= market_moments
        end

        return [Vector(all_moments[i, :]) for i in 1:S]
    end
end #@everywhere


function save_dataset(dataset::DataFrame, filename::String; folder_path = nothing)
    if isnothing(folder_path)
        folder_path = tempdir()
    end
    file_path = joinpath(folder_path, filename)
    CSV.write(file_path, sort!(dataset))
    return file_path
end

function data_folder_path()
    return joinpath(_ROOT_DIR, _DATA_FOLDER_NAME)
end
function replication_data_folder_path()
    return joinpath(_ROOT_DIR, _DATA_FOLDER_NAME, _REPLICATION_DATA_FOLDER)
end
function ensure_folder_exists(path)
    if isdir(path)
        return path
    else
        try
            mkdir(path) 
            return path
        catch e
            print("\nAn error while trying to create folder: \n", 
                "\nTarget Folder Path: '$path'\n",
                "\nError: $e")
            error(e)
        end
    end
end
function ensure_data_folder_exists()
    return ensure_folder_exists(data_folder_path())
end
function ensure_replication_data_folder_exists()
    return ensure_folder_exists(replication_data_folder_path())
end
function replication_data_path(r::Int)
    return joinpath(replication_data_folder_path(), "replication_$r.csv")
end 
function replication_data_exists(r::Int)
    return isfile(replication_data_path(r))
end
function load_replication_data(r::Int)
    if replication_data_exists(r)
        path = replication_data_path(r)
        println("reading replication data from file '$path'")
        return  DataFrame(CSV.File(path))
    else
        error("Cannot load replication data - file not found")
    end
end
function save_replication_data(data, r::Int)
    ensure_replication_data_folder_exists()
    path = replication_data_path(r)
    CSV.write(replication_data_path(r), sort!(data))
    return path
end

function simulate_data(theta::Vector{Float64}, S::Int64, T::Int64, N::Int64, r::Int64, model::Model, H::Matrix{VariableRef})
    rho1::Float64 = theta[1]
    rho2::Float64 = theta[2]
    rho3::Float64 = theta[3]
    sigma::Float64 = theta[4]
    gamma::Vector{Float64} = theta[5:8]

    # Simulate data for given theta with S simulations
    sigma_B = generate_covariance(N, sigma, rho1, rho2, rho3)
    sigma_B = (sigma_B + sigma_B') / 2
    chol_sigma_B = nothing
    try
        chol_sigma_B = cholesky(sigma_B; check=true)
    catch
        return DataFrame()
    end 

    tasks = [(s, t) for s in 1:S for t in 1:T]

    @everywhere begin
        CHOL_SIGMA_B = $chol_sigma_B
        GAMMA = $gamma
        K = length(GAMMA)
        N = $N
        r = $r
        model = $model
        H = $H
    end
    
    @everywhere begin
        function simulate_market_task(task)
            s = task[1]
            t = task[2]
            return generate_market_simluation(r, s, t, N, K, GAMMA, CHOL_SIGMA_B, model, H) 
        end
    end
    results = pmap(simulate_market_task, tasks)
    return vcat(results...)
end

function msm_objective(theta::Vector{Float64}, replication_moments::Vector{Vector{Float64}}, 
        W::Matrix{Float64}, N::Int64, T::Int64, S::Int64, r::Int64, model::Model, H::Matrix{VariableRef})

    rho2::Float64 = theta[2]
    rho3::Float64 = theta[3]
    sigma::Float64 = theta[4]
    gamma::Vector{Float64} = theta[5:8]

    simulated_data = simulate_data(theta, S, T, N, r, model, H)

    # Check if simulated_data is empty (due to non-positive definite Sigma_B)
    if nrow(simulated_data) == 0
        return 1e10  # Return a large value for the objective function
    end

    running_market_sum = zeros(_NUM_MOMENTS)

    for (t, cross_simulation_market_data) in enumerate(groupby(simulated_data, :market_id))
        market_sim_moments = compute_all_simulation_moments_for_market(cross_simulation_market_data)
        temp_sum = zeros(_NUM_MOMENTS)
        for s in 1:S
            temp_sum .+= market_sim_moments[s] - replication_moments[t]
        end
        running_market_sum .+= temp_sum ./ S
    end
    total_avg_diff = running_market_sum ./ T
    
    Q = dot(total_avg_diff, W * total_avg_diff)
    return Q
end

function estimate_theta(data::DataFrame, W::Matrix{Float64}, 
        theta_initial::Vector{Float64}, N::Int64, T::Int64, S::Int64, 
        r::Int64, optimization_settings::Dict, model::Model, H::Matrix{VariableRef})

    theta = theta_initial
    replication_moments = compute_all_markets_moments(data)
    obj(theta) = msm_objective(theta, replication_moments, W, N, T, S, r, model, H)
    res = bboptimize(
        obj; optimization_settings...)
    theta_hat = best_candidate(res)
    solver_info = Dict(
        "stop_reason" => BlackBoxOptim.stop_reason(res),       
        "best_fitness" =>  BlackBoxOptim.best_fitness(res),
        "best_candidate" => BlackBoxOptim.best_candidate(res),
    )
    return theta_hat, solver_info
end

function generate_replication_data(R::Int64, N::Int64, T::Int64, gamma::Vector{Float64}, 
    sigma::Float64, rho1::Float64, rho2::Float64, rho3::Float64, model::Model, H::Matrix{VariableRef})

    sigma_B = generate_covariance(N, sigma, rho1, rho2, rho3)
    sigma_B = (sigma_B + sigma_B') / 2
    chol_sigma_B = nothing
    try
        chol_sigma_B = cholesky(sigma_B; check=true)
    catch
        error("Covariance matrix sigma_B is not positive definite.")
    end

    tasks = [(r, t) for r in 1:R for t in 1:T]

    @everywhere begin
        const CHOL_SIGMA_B = $chol_sigma_B
        const GAMMA = $gamma
        const K = length(GAMMA)
        const N = $N
        const T = $T
        model = $model
        H = $H
    end

    @everywhere begin
        function generate_market_replication_task(task)
            r = task[1]
            t = task[2]
            return generate_market_replication(r, t, N, K, GAMMA, CHOL_SIGMA_B, model, H) 
        end
    end
    results = pmap(generate_market_replication_task, tasks)
    return vcat(results...)
end

function present_results(settings::Dict{String, Any}, theta_true::Vector{Float64}, bias::Vector{Float64},
    rmse::Vector{Float64}, coverage::Vector{Float64}, h::Vector{Float64}, weighting_matrix_type::String;
    output_file::String = "", append::Bool = false)
    
    # Parameter names excluding the normalized Gamma 1
    parameter_names = ["Rho 1", "Rho 2", "Rho 3", "Sigma", "Gamma 1", "Gamma 2", "Gamma 3", "Gamma 4"]

    if output_file == ""
        output_file = "tables_R$(settings["R"])_S$(settings["S"])_T$(settings["T"])_N$(settings["N"])_MS$(settings["MaxSteps"])_PS$(settings["PopulationSize"])_$weighting_matrix_type.tex"
    end 

    # Create the settings table
    settings_table = DataFrame(
        Setting = ["Number of Replications",
            "Number of Simulations",
            "Number of Markets",
            "Market Size",
            "Number of Moments",
            "Optimization Method",
            "Max Steps",
            "Max Time",
            "Population Size"],

        Value = Any[
            settings["R"],
            settings["S"],
            settings["T"],
            settings["N"],
            settings["NumberOfMoments"],
            settings["Method"],
            settings["MaxSteps"],
            settings["MaxTime"],
            settings["PopulationSize"],
            ]
    )

    results_table = DataFrame(
        "Parameter Name" => parameter_names,
        "True Parameters" => theta_true,
        "Bias" => bias,
        "RMSE" => rmse,
        "Coverage (95%)" => coverage,
        "Step Size" => h,
    )

    function write_table(io::IO, table::DataFrame, title::String)
        write(io, "\\subsection*{" * title * "}\n\n")
        pretty_table(io, table; backend=Val(:latex))
        write(io, "\n\n")
    end

    file_mode = append ? "a" : "w"
    open(output_file, file_mode) do io
        write(io, "\\section*{Monte Carlo Results}\n\n")
        write_table(io, settings_table, "Settings")
        write_table(io, results_table, "Estimation Results with $weighting_matrix_type Weighting Matrix")
    end
    println("\nTables have been saved to '$output_file'")
end

function compute_optimal_weighting_matrix(replication_data)
    small_delta = 1e-6
    replication_moments = compute_all_markets_moments(replication_data)
    Omega_hat = cov(replication_moments, corrected=false)
    Omega_hat += small_delta * I(size(Omega_hat, 2))
    W = inv(Omega_hat)
    return W
end

function compute_avg_moments(data::DataFrame, S::Int64, T::Int)
    running_market_sum = zeros(_NUM_MOMENTS)
    for (t, cross_simulation_market_data) in enumerate(groupby(data, :market_id))
        market_sim_moments = compute_all_simulation_moments_for_market(cross_simulation_market_data)
        temp_sum = zeros(_NUM_MOMENTS)
        for s in 1:S
            temp_sum .+= market_sim_moments[s]
        end
        running_market_sum .+= temp_sum ./ S
    end
    return running_market_sum ./ T
end 

function compute_variance_matrix(theta_hat::Vector{Float64}, data::DataFrame, W::Matrix{Float64}, 
        S::Int64, T::Int64, N::Int64, r::Int64, h::Vector{Float64}, model::Model, H::Matrix{VariableRef})
    num_params = length(theta_hat)
    num_moments = _NUM_MOMENTS

    G = zeros(num_moments, num_params)

    for i in 1:num_params
        theta_plus = copy(theta_hat)
        theta_minus = copy(theta_hat)
        theta_plus[i] += h[i]
        theta_minus[i] -= h[i]

        data_plus = simulate_data(theta_plus, S, T, N, r, model, H)
        data_minus = simulate_data(theta_minus, S, T, N, r, model, H)
        #if nrow(simulated_data) == 0
        moments_plus = compute_avg_moments(data_plus, S, T)
        moments_minus = compute_avg_moments(data_minus, S, T)
        
        G[:, i] = (moments_plus - moments_minus) / (2 * h[i])
    end

    moments_T = zeros(T, num_moments)
    for t in 1:T
        market_data = data[data.market_id .== t, :]
        moments_T[t, :] = compute_market_moments(market_data)
    end
    omega = cov(moments_T, dims=1, corrected=false) + 1e-6 * I(num_moments)
    
    factor = (1 + 1/S) / T

    GWG = G' * W * G
    inv_GWG = pinv(GWG)

    AVAR = factor * inv_GWG * (G' * W * omega * W * G) * inv_GWG
    
    # Compute standard errors
    std_errors = sqrt.(diag(AVAR))
    
    z_value = quantile(Distributions.Normal(0, 1), 0.975)
    lower_bounds = theta_hat .- z_value .* std_errors
    upper_bounds = theta_hat .+ z_value .* std_errors
    
    confidence_intervals = hcat(lower_bounds, upper_bounds)
    
    return std_errors, confidence_intervals
end

function compute_statistics(theta_estimates::Array{Float64, 2}, confidence_intervals::Array{Float64, 3}, theta_true::Vector{Float64})
    num_params = length(theta_true)
    bias::Vector{Float64} = vec(mean(theta_estimates, dims=1) - theta_true')
    rmse::Vector{Float64} = vec(sqrt.(mean((theta_estimates .- theta_true').^2, dims=1)))
    
    # Compute coverage
    coverage = zeros(num_params)
    for i in 1:num_params
        lower_bounds = confidence_intervals[:, i, 1]
        upper_bounds = confidence_intervals[:, i, 2]
        coverage[i] = mean((theta_true[i] .>= lower_bounds) .& (theta_true[i] .<= upper_bounds))
    end
    
    return bias, rmse, coverage
end

function main(R::Int64, N::Int64, T::Int64, S::Int64, theta::Vector{Float64}; regenerate_replication_data::Bool = false, optimal_weights::Bool = true)
    rho1::Float64 = theta[1]
    rho2::Float64 = theta[2]
    rho3::Float64 = theta[3]
    sigma::Float64 = theta[4]
    gamma::Vector{Float64} = theta[5:8]


    optimization_settings = Dict(
        :Method => :adaptive_de_rand_1_bin_radiuslimited,
        :SearchRange => collect(zip(_SEARCH_RANGE_LB, _SEARCH_RANGE_UB)),
        :MaxSteps => _BBOPTIMIZE_MAX_STEPS,
        :MaxTime => _BBOPTIMIZE_MAX_TIME,
        :TraceMode => :compact,
        :PopulationSize => _BBOPTIMIZE_POPULATION_SIZE,
        :NumDimensions => length(_THETA_0)
    )

    model, H = create_matching_model(N)

    if (regenerate_replication_data) || !isdir(replication_data_folder_path())
        println("generating replication data")
        replication_data = generate_replication_data(R, N, T, gamma, sigma, rho1, rho2, rho3, model, H)
        for r in 1:R
            path = save_replication_data(replication_data[replication_data.replication_id .== r, :], r)
            println("replication $r saved to file '$path'")
        end
        println("done with replication data")
    else
        println("NOT generating replication data -- will load existing replications on demand")
    end


     all_solver_info = DataFrame()
     W = I(_NUM_MOMENTS)
     theta_true = theta
     theta_estimates = zeros(R, length(theta_true))
     std_errors = zeros(R, length(theta_true))
     confidence_intervals = Array{Float64, 3}(undef, R, length(theta_true), 2)

     h = [1e-5 for _ in 1:length(theta_true)]
     #current_theta = initial_theta 
     
     for r in 1:R
        rep_data = load_replication_data(r)

        Random.seed!(5678 + r)
        
        if optimal_weights
            W = compute_optimal_weighting_matrix(rep_data)
        else
            W = I(_NUM_MOMENTS)
        end

        theta_initial = theta_true .+ randn(length(theta_true)) * 0.1 

        #theta_hat, solver_info = estimate_theta(data, W, theta0, N, T, S, r)
        
        theta_hat, solver_info = estimate_theta(rep_data, W, theta_initial, N, T, S, r, optimization_settings, model, H)
        
        std_errs, conf_intervals = compute_variance_matrix(theta_hat, rep_data, W, S, T, N, r, h, model, H)
        
        append!(all_solver_info, DataFrame(solver_info))
        theta_estimates[r, :] = theta_hat
        std_errors[r, :] = std_errs
        confidence_intervals[r, :, :] = conf_intervals
     end

     bias, rmse, coverage = compute_statistics(theta_estimates, confidence_intervals, theta_true)
     results_table_settings = Dict(
        "R" => R, 
        "S" => S,
        "T" => T,
        "N" => N,
        "NumberOfMoments" => _NUM_MOMENTS,
        "Method" => optimization_settings[:Method],
        "MaxSteps" => optimization_settings[:MaxSteps],
        "MaxTime" => optimization_settings[:MaxTime],
        "PopulationSize" => optimization_settings[:PopulationSize],
     )
     
     weight_matrix_type = "Identity"
     if optimal_weights
        weight_matrix_type = "Optimal"
     end

     present_results(results_table_settings, theta_true, bias, rmse, coverage, h, weight_matrix_type)
     CSV.write(joinpath(_ROOT_DIR, "solver_info.csv"), all_solver_info)

     println("========================")
     println("THETA0")
     println(_THETA_0)
     println("========================")
     println("THETA ESTIMATES")
     for theta_estimate in theta_estimates
        println(theta_estimate)
     end 
     println("========================")
end #main


println("start")
println("Running with identity weighting matrix")
time_taken = @elapsed main(_R, _N, _T, _S, _THETA_0,
    regenerate_replication_data = _REGENERATE_REPLICATION_DATA, optimal_weights = false)
println("time taken = ", time_taken)
println("Running with optimal weighting matrix")
time_taken = @elapsed main(_R, _N, _T, _S, _THETA_0,
    regenerate_replication_data = _REGENERATE_REPLICATION_DATA, optimal_weights = true)
println("time taken = ", time_taken)
println("end")



