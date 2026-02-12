using JuMP
import Ipopt

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
_BETA_1 = 5
_BETA_2 = 6
_ALPHA_11 = -1
_ALPHA_12 = 0.5
_ALPHA_21 = -0.75
_ALPHA_22 = -1
_C_1 = 2
_C_2 = 1
_A_11 = 1
_A_12 = 2
_A_21 = 2
_A_22 = 3
_B_1 = 5
_B_2 = 8


function initModel()
#------------------------------------------------------------------------------
# Initialize model around decision variables, objectives and constraints. In 
# order of appearance below we define... 
#   1) Price decision variables (p1, p2), include non-negative constraints
#   2) Demand equations as described in question
#   3) Profit function (Revenue - Cost)
#   4) The resource constraints
#------------------------------------------------------------------------------
    model = Model(Ipopt.Optimizer)

    @variable(model, p1 >= 0)
    @variable(model, p2 >= 0)

    @expression(model, x1, _BETA_1 + _ALPHA_11 * p1 + _ALPHA_12 * p2)
    @expression(model, x2, _BETA_2 + _ALPHA_21 * p1 + _ALPHA_22 * p2)

    @objective(model, Max, p1 * x1 + p2 * x2 - (_C_1 * x1 + _C_2 * x2))

    @constraint(model, _A_11 * x1 + _A_12 * x2 <= _B_1)
    @constraint(model, _A_21 * x1 + _A_22 * x2 <= _B_2)
    
    return model, p1, p2, x1, x2
end 

function prettyPrintResults(model, p1, p2, x1, x2)
#------------------------------------------------------------------------------
# Not a functionally interesting bit, just printing semi-readable results
#------------------------------------------------------------------------------
    println("-----------------------------")
    println("          Results            ")
    println("- - - - - - - - - - - - - - -")
    println("+ Optimal Quantity")
    println("  Diaper type 1 = ", value(x1))
    println("  Diaper type 2 = ", value(x2))
    println("- - - - - - - - - - - - - - -")
    println("+ Optimal Prices")
    println("  Diaper type 1 = ", value(p1))
    println("  Diaper type 2 = ", value(p2))
    println("- - - - - - - - - - - - - - -")
    println("+ Optimal Resource use")
    println("  Fabric 1 in diaper 1 = ", value(_A_11) * value(x1))
    println("  Fabric 1 in diaper 2 = ", value(_A_12) * value(x2))
    println("  Total amount of Fabric 1 used = ", value(_A_11) * value(x1) + value(_A_12) * value(x2))
    println("  ")
    println("  Fabric 2 in diaper 1 = ", value(_A_21) * value(x1))
    println("  Fabric 2 in diaper 2 = ", value(_A_22) * value(x2))
    println("  Total amount of Fabric 2 used = ", value(_A_21) * value(x1) + value(_A_22) * value(x2))
    println("- - - - - - - - - - - - - - -")
    println("Optimal profit = ", objective_value(model))
    println("-----------------------------")
    return nothing
end
    
function main()
#------------------------------------------------------------------------------
# Conductor
#------------------------------------------------------------------------------
    model, p1, p2, x1, x2 = initModel()

    # Solve the model
    optimize!(model)
    @assert is_solved_and_feasible(model)
    
    prettyPrintResults(model, p1, p2, x1, x2)

    return nothing
end

main()