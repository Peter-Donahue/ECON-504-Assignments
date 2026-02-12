using JuMP
import HiGHS

_CHAIRS = "Chairs"
_DESKS  = "Desks "
_TABLES = "Tables"

struct ProductInputs
    fabrication::Float64
    assembly::Float64
    machining::Float64
    wood::Float64
    profit::Float64
end

struct Resources
    fabrication::Float64
    assembly::Float64
    machining::Float64
    wood::Float64
end

function initData()::Tuple{Dict{String, ProductInputs},Resources}
#------------------------------------------------------------------------------
# Initialize a dictionary of name-keyed ProductInputs structs with the data on
# unit costs and profit for each and a Resources struct
#------------------------------------------------------------------------------
    return Dict(_CHAIRS => ProductInputs(1, 3, 9, 6, 4)
                , _DESKS => ProductInputs(3, 8, 6, 4, 15)
                , _TABLES => ProductInputs(1, 6, 4, 5, 14)), 
            Resources(1000, 2000, 1440, 1440)
end


function initModel(inputs::Dict{String, ProductInputs}, resources::Resources)::Tuple{Model,Any}
#------------------------------------------------------------------------------
# Creates Optimizer, builds model st:
#   1) variables are products (Chairs, Tables, and Desks)
#   2) objective is to maximized profit (unit profit * number of units) 
#   3) Non-negativity - number of units must be >= for all products
#   4) Resources contraints - units of output for product `x` * units of resource 
#      `y` required per unit `x` must be
#       - non-negative 
#       - less than or equal to the total units of `y` available
#------------------------------------------------------------------------------
    model = Model(HiGHS.Optimizer)
    
    productNames = keys(inputs)
    profitPerUnit = [inputs[name].profit for name in productNames]

    @variable(model, products[productNames])
    @objective(model, Max, sum(profitPerUnit .*  Array(products)));

    # Non-Negativity Constraint
    @constraint(model, 0 <= Array(products))

    #Resource Constraint
    @constraint(
        model,
        [resource in fieldnames(typeof(resources))],
        sum([getfield(inputs[name], resource) for name in productNames] .* Array(products)) 
        <=  getfield(resources, resource),
    )

    return model, products
end


function prettyPrintResults(inputs, resources, model, products)
#------------------------------------------------------------------------------
# Not a functionally interesting bit, just printing semi-readable results
#------------------------------------------------------------------------------
    println("-----------------------------")
    println("          Results            ")
    println("- - - - - - - - - - - - - - -")
    println("+ Units Produced")
    totalProfit = 0.0
    resourcesUsed = Dict()
    for resource in fieldnames(typeof(resources))
        resourcesUsed[resource] = 0.0
    end 

    for productName in keys(inputs)
        qty = value(products[productName])

        println("  ", productName, " = ", qty)
        
        totalProfit += qty*inputs[productName].profit

        for resource in keys(resourcesUsed)
            resourcesUsed[resource] += qty*getfield(inputs[productName], resource) 
        end
    end

    println("- - - - - - - - - - - - - - -")
    println("+ Resource: Used / Available")
    for resource in fieldnames(typeof(resources))
        println("  ", resource, ": ", resourcesUsed[resource], " / ", getfield(resources, resource))
    end
    println("- - - - - - - - - - - - - - -")
    println("Total Profit = \$", totalProfit)
    println("-----------------------------")
    return nothing
end


function main()
#------------------------------------------------------------------------------
# Conductor
#------------------------------------------------------------------------------
    inputs, resources = initData()
    model, products = initModel(inputs, resources)
    
    optimize!(model)
    @assert is_solved_and_feasible(model)

    prettyPrintResults(inputs, resources, model, products)
end

main()

