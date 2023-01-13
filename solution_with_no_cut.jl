using CSV
using DataFrames
using JuMP
using GLPK
using Printf


cd("D:\\case_study")
pwd()
s = Matrix(CSV.read("s.csv", DataFrame))
zᵖ = Matrix(CSV.read("z_p.csv", DataFrame) )
zᶜ = Matrix(CSV.read("z_c.csv", DataFrame) )
zᵐ = Matrix(CSV.read("z_m.csv", DataFrame) )
D = Matrix(CSV.read("D.csv", DataFrame) )



H=H
J=J
T=T
Ω=Ω
D = rand.(D, Ω)
p = [1/Ω for ω in 1:Ω]

flexibility = flexibility


m = Model(GLPK.Optimizer)
@variable(m , ψ[1:H], Bin)
@variable(m , χ[1:H , 1:J], Bin)
@variable(m , α[1:H , 1:J , 1:T , 1:Ω], Bin);
@variable(m , 0 ≤ γ[1:J , 1:T , 1:Ω], Int )


@objective(m , Min , 
    sum(T * zᵖ[h] * ψ[h] for h in 1:H ) + 
    sum(zᵐ[h,j] *χ[h,j] for h in 1:H for j in 1:J ) +
    sum( p[ω] * zᶜ[j] * γ[j,t,ω]  for j in 1:J for t in 1:T for ω in 1:Ω )
)


c1 = @constraint(m, [h in 1:H , j in 1:J], χ[h,j] ≤ ψ[h]  )
cc = @constraint(m, [h in 1:H , j in 1:J], s[h,j] + ψ[h] ≤ χ[h,j] + 1  )
c2 = @constraint(m, [h in 1:H , j in 1:J , t in 1:T , ω in 1:Ω ], α[h,j,t,ω] ≤ χ[h,j] )
c3 = @constraint(m, [h in 1:H , t in 1:T , ω in 1:Ω], sum( α[h,j,t,ω] for j in 1:J) ≤ 1 )
c4 = @constraint(m, [j in 1:J , t in 1:T , ω in 1:Ω], sum( α[h,j,t,ω] for h in 1:H) + γ[j,t,ω] ≥ D[j,t,ω]);

# extra constraints
c5 = @constraint(m, [h in 1:H], sum(χ[h,j] for j in 1:J) ≤ flexibility )

optimize!(m)

solution_summary(m)


χᵏ = value.(χ)
ψᵏ = value.(ψ)
γᵏ = value.(γ)
no_permanent = sum(ψᵏ)
no_skill = sum(χᵏ)
no_multiskilling = no_skill - no_permanent
cost_permanent = (sum(T * zᵖ[h] * ψᵏ[h] for h in 1:H ) )/T
cost_multiskillling = sum(zᵐ[h,j] *χᵏ[h,j] for h in 1:H for j in 1:J ) / 1_000_000
cost_cacual = (sum( p[ω] * zᶜ[j] * γᵏ[j,t,ω]  for j in 1:J for t in 1:T for ω in 1:Ω ) )/ (T * 1_000_000)
cost_total = (objective_value(m) )/T

println("******************************************************")
println("************** solution ******************************")

@show no_permanent
@show no_skill
@show no_multiskilling
@show cost_total
@show cost_permanent
@show cost_multiskillling
@show cost_cacual;
solution_summary(m)
@show num_constraints(m; count_variable_in_set_constraints = false)
@show num_variables(m)

dt=DataFrame(no_per=no_permanent,no_skill=no_skill,no_multi=no_multiskilling , cost_total=cost_total ,
cost_permanent=cost_permanent , cost_multi = cost_multiskillling , cost_cas = cost_cacual
)

