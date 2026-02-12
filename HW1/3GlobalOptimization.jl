using Distributions
using BlackBoxOptim
# == Plot Generation
#using Plots
#using Base.Threads
# ==

# Constants
_N = 10_000
_TRUE_BETA = 2.0
_X_DIST_MU = [-0.7, 1.0]
_X_DIST_COV = [1.0 0.1; 0.1 0.5]

_SEARCH_RANGE = [(-5.0, 5.0)]
_MAX_STEPS = 10
_POPULATION_SIZE = 50
_BETA_STEP = 0.1

# == Plot Generation
#_GRAPH_OUTPATH = "C:\\Users\\peter\\Git\\ECON504\\HW1\\code\\plot.png"
# ==

#------------------------------------------------------------------------------
# Generate data on x1, x2, epsilon, and y
# Global variables... yes, lazy, my bad...
#------------------------------------------------------------------------------
x = rand(MvNormal(_X_DIST_MU, _X_DIST_COV), _N)'
x1 = x[:, 1]
x2 = x[:, 2]
epsilon = [rand(Normal(0, abs(0.3 * x1[i] + 0.8 * x2[i]))) for i in 1:_N]    
y = @. (x1 + x2 * _TRUE_BETA + epsilon) >= 0


function max_score_objective(beta)
    correct_predictions = sum((y .== 1) .* ((x1 .+ x2 .* beta) .>= 0) .+ (y .== 0) .* ((x1 .+ x2 .* beta) .< 0))
    return correct_predictions / _N
end

function objective_wrapper(beta_vec)
    beta = beta_vec[1]
    return -max_score_objective(beta)  # Minimizing, so we return the negative
end

# == Plot Generation
# function plotResults(optimized_beta)
#    betas = _SEARCH_RANGE[1][1]:_BETA_STEP:_SEARCH_RANGE[1][2]
   
#    objective_values = Vector{Float64}(undef, length(betas))  
#    @threads for i in 1:length(betas)
#        objective_values[i] = max_score_objective(betas[i])
#    end

#    plot(betas, objective_values, xlabel="Beta", ylabel="Objective Function Value", label="Q(beta)", title="Maximum Score Objective Function", legend=true)
#    scatter!([optimized_beta], [max_score_objective(optimized_beta)], label="Optimized beta", color=:red)

#    p = plot(betas, objective_values, xlabel="Beta", ylabel="Objective Function Value", label="Q(beta)", title="Maximum Score Objective Function", legend=true)
#    scatter!([optimized_beta], [max_score_objective(optimized_beta)], label="Optimized beta", color=:red)

#    savefig(p, _GRAPH_OUTPATH) 
# end
# == 

function main()
#------------------------------------------------------------------------------
# Conductor
#------------------------------------------------------------------------------
    results = bboptimize(objective_wrapper; 
        SearchRange = _SEARCH_RANGE, 
        MaxSteps = _MAX_STEPS, 
        PopulationSize = _POPULATION_SIZE
    )

# == Plot Generation
#    plotResults(results.archive_output.best_candidate[1])
# ==
    return nothing
end

main()