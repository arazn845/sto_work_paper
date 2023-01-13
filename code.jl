using Gurobi
using JuMP
using CSV, DataFrames
using Distributions, Random

##########################################
# parameters
##########################################

s = Matrix(CSV.read("s.csv", DataFrame))
zᵖ = Matrix(CSV.read("z_p.csv", DataFrame) )
zᶜ = Matrix(CSV.read("z_c.csv", DataFrame) )
zᵐ = Matrix(CSV.read("z_m.csv", DataFrame) )
D = Matrix(CSV.read("D.csv", DataFrame) )

R = R
H = H
J = J
Ω = Ω


p = [1/Ω for ω in 1:Ω]

################################################################
# index_generator
################################################################

function index_generator(χᵏ)
    Hᵏ = [ [] for j in 1:J ] # for 1
    Hᵖ = [ [] for j in 1:J ] # for 0
    Hᵀ = [h for h in 1:H]

        for j in 1:J
        
            for h in 1:H
                if χᵏ[h,j] == 1
                    push!(Hᵏ[j] , h)
                end
            end
        end

        for j in 1:J
            Hᵖ[j] = setdiff( Hᵀ , Hᵏ[j] )
        end
   Jᵀ = [j for j in 1:J]
    Jᵏ = [ [] for h in 1:H]
    Jᵛ = [ [] for h in 1:H]
    for h in 1:H
        for j in 1:J
            if χᵏ[h,j] == 1
                push!(Jᵏ[h] , j)
            end
        end
    end
    for h in 1:H    
        Jᵛ[h] = setdiff(Jᵀ , Jᵏ[h])
    end
    return (Hᵏ , Hᵖ , Jᵏ , Jᵛ)
end
#######################################################################################
# f_zʳ
#######################################################################################
function f_zʳ(γᵏ) 
    s = [ [] for t in 1:T, ω in 1:Ω]
    for t in 1:T
        for ω in 1:Ω
            append!(s[t,ω] , [ones(Int(γᵏ[j,t,ω]) )  * zᶜ[j] for j in 1:J] )
            s[t,ω] = reduce(vcat , s[t,ω])
            s[t,ω] = sort(s[t,ω] , rev=true)
            if length(s[t,ω]) < H
                append!( s[t,ω] , [0.0 for i in 1:(H - length(s[t,ω]) ) ] )
            end
        end
    end
    return  s
end

#######################################################################################
# f_dʳ
#######################################################################################
function f_dʳ(zʳ)    
    
    d=[fill(0.0 , H) for t in 1:T, ω in 1:Ω]
    for t in 1:T
        for ω in 1:Ω
            for k in 1:H
                d[t,ω][k]= sum( zʳ[t,ω][l] for l in 1:k)
            end
        end
    end
    return d
end

#######################################################################################
# main problem
#######################################################################################
mp = Model(Gurobi.Optimizer)
@variable(mp, ψ[1:H], Bin)
@variable(mp, χ[1:H , 1:J], Bin)
@variable(mp, 0 ≤ β[1:T , 1:Ω])
# auxilary variable #########################


@objective(mp, Min, 
    sum( T*zᵖ[h]*ψ[h] for h in 1:H ) +
    sum(χ[h,j] * zᵐ[h,j] for h in 1:H for j in 1:J) +
    sum( p[ω] * β[t,ω] for t in 1:T for ω in 1:Ω )
)
c1 = @constraint(mp, [h in 1:H , j in 1:J], χ[h,j] ≤ ψ[h])
c2 = @constraint(mp, [h in 1:H , j in 1:J], s[h,j] + ψ[h] ≤ χ[h,j] + 1);


#######################################################################################
# sub problem function
#######################################################################################
function solve_sub_prob(χʳ)
    sp =[ Model(Gurobi.Optimizer) for t in 1:T, ω in 1:Ω]
    θ_1 = fill(0.0,T,Ω)
    γ_1 = fill(0.0,J,T,Ω)
    for v in 1:T
        for u in 1:Ω
            @variable(sp[v,u], 0 ≤ γ[1:J , [v] , [u] ], Int )
            @variable(sp[v,u], α[1:H , 1:J, [v] , [u] ], Bin )
            @objective(sp[v,u], Min, sum(zᶜ[j] * γ[j,v,u] for j in 1:J) )
            @constraint(sp[v,u], con_23b[h in 1:H , j in 1:J], α[h,j,v,u] ≤ χʳ[h,j] )
            @constraint(sp[v,u], con_23c[h in 1:H], sum( α[h,j,v,u] for j in 1:J) ≤ 1 )
            @constraint(sp[v,u], con_23d[j in 1:J], sum( α[h,j,v,u] for h in 1:H ) + γ[j,v,u] ≥ D[j,v,u] )
            optimize!(sp[v,u])
            θ_1[v,u] = objective_value(sp[v,u])
            γ_1[:,v,u] = round.(value.(γ[:,v,u]) , digits=1)
        end
    end
    
    return θ_1, γ_1
end

#######################################################################################
# initiating Benders cut
#######################################################################################

function initiate()
    set_r=[]
    set_cost_per=[]
    set_cost_train=[]
    set_βʳ=[]
    set_cost_cas=[]
    set_time=[]
    set_LB=[]
    set_UB=[]
    set_Θ_min=[]
    set_gap=[]
    set_gap_min=[]
    set_noVar = []
    set_noCon = []
    for r in 1:R
        println("          main   ##############################")
        optimize!(mp)
        ψʳ = value.(ψ)
        χʳ = value.(χ)
        βʳ = value.(β)
        cost_per = sum( T*zᵖ[h]*ψʳ[h] for h in 1:H ) 
        cost_train = sum(χʳ[h,j] * zᵐ[h,j] for h in 1:H for j in 1:J) 
        βʳ = sum( p[ω] * βʳ[t,ω] for t in 1:T for ω in 1:Ω )
        println("          index   ##############################")
        ig = index_generator(χʳ)
        Hʳ = ig[1]
        Hᵖ = ig[2]
        Jʳ = ig[3]
        Jᵖ = ig[4]
        println("          sub   ##############################")
        ss = solve_sub_prob(χʳ)
        θʳ=ss[1]
        γʳ=ss[2]
        cost_cas = sum( p[ω] * θʳ[t,ω] for t in 1:T for ω in 1:Ω)
        println("          z   ##############################")
        zʳ=f_zʳ(γʳ)
        println("          d   ##############################")
        #@show value.(χ)
        #for t in 1:T 
        #    for ω in 1:Ω
        #        @show zʳ[t,ω]
        #    end
        #end
        l = [length(zʳ[t,ω]) for t in 1:T, ω in 1:Ω]
        #@show l
        for t in 1:T
            for ω in 1:Ω
                if length(zʳ[t,ω]) < H
                    push!(zʳ[t,ω] , 0.0)
                end
            end
        end
        #for t in 1:T 
        #    for ω in 1:Ω
        #        @show zʳ[t,ω]
        #    end
        #end
        dʳ=f_dʳ(zʳ)
        #################################
        println("")
        println("")
        println("          χ   ##############################")
        dt_χʳ = DataFrame(χʳ , :auto)
        
        #@show Hʳ
        #@show Hᵖ
        #@show Jʳ
        #@show Jᵖ
        
        @show dt_χʳ
        #if r > 1
        #    cʳ = value.(c)
        #    dt_cʳ = DataFrame(cʳ[:,:,1] , :auto)
        #    @show dt_cʳ
        #    aʳ = value.(a)
        #    dt_aʳ = DataFrame( aʳ , :auto )
        #    @show dt_aʳ
        #    bʳ = value.(b)
        #    dt_bʳ = DataFrame( bʳ , :auto )
        #    @show dt_bʳ
        #end
        println("")
        println("")
        println("          zʳ   ##############################")
        #for t in 1:T 
        #    for ω in 1:Ω
        #        @show zʳ[t,ω]
        #    end
        #end
        println("")
        println("")
        #for t in 1:T 
        #    for ω in 1:Ω
        #        @show dʳ[t,ω]
        #    end
        #end
        println("")
        println("")
        dt_θʳ = DataFrame(θʳ , :auto)
        #@show dt_θʳ
        push!(set_r , r)
        push!(set_cost_per, cost_per)
        push!(set_cost_train, cost_train)
        push!(set_βʳ, βʳ)
        push!(set_cost_cas, cost_cas)
        
        push!(set_time , solve_time(mp) )
        LB = objective_value(mp)
        push!(set_LB , LB)
        UB = cost_per + cost_train + cost_cas
        push!(set_UB , UB)
        Θ_min = minimum(set_UB)
        push!(set_Θ_min , Θ_min)
        gap = (UB - LB) / UB
        push!(set_gap , gap)
        gap_min = minimum(set_gap)
        push!(set_gap_min , gap_min)
        push!(set_noVar , num_variables(mp) )
        push!(set_noCon , num_constraints(mp; count_variable_in_set_constraints = false) )
        println("")
        println("")
        dt1 = DataFrame( r = set_r , cost_per = set_cost_per , cost_train = set_cost_train , 
            βʳ = set_βʳ , cost_cas = set_cost_cas , noVar = set_noVar , noCon = set_noCon)
        @show dt1
        println("")
        println("")
        dt2 = DataFrame( r = set_r , time = set_time , LB = set_LB , UB = set_UB , 
            Θ_min = set_Θ_min , gap = set_gap , gap_min = set_gap_min )
        @show dt2
        if gap_min ≤ 0.001
            println("##################################")
            println("########## optimality      #######")
            @show sum(set_time)
            break
        end
         b = @variable(mp, [1:H ], Bin)
        c = @variable(mp, [1:T , 1:Ω])
        @constraint(mp, [t in 1:T , ω in 1:Ω], c[t,ω] ≥ 0 )
        a = @variable(mp, [1:H], Bin)
        
       
        c_29a = @constraint(mp, [t in 1:T , ω in 1:Ω], β[t,ω] ≥ θʳ[t,ω] - c[t,ω] - 
            θʳ[t,ω] * sum( (1 - χ[h,j] ) for j in 1:J for h in Hʳ[j] )   )
        c_29b = @constraint(mp, [h in 1:H], J * b[h] ≥ sum(χ[h,j] for j in Jᵖ[h]) )  
        c_29c = @constraint(mp, [h in 1:H], b[h] ≤ sum( χ[h,j] for j in Jᵖ[h] ) ) 
        
        c_29d = @constraint(mp, [t in 1:T , ω in 1:Ω], c[t,ω] ≤ sum( a[k]*dʳ[t,ω][k] for k in 1:H ) )
        c_29e = @constraint(mp, [t in 1:T , ω in 1:Ω], c[t,ω] ≤ sum( zʳ[t,ω][i] for i in 1:H ) )
        c_29f = @constraint(mp, [k in 1:H], sum( k * a[k] for k in 1:H) == sum(b[h] for h in 1:H) )
        c_29g = @constraint(mp, sum( a[k] for k in 1:H ) == 1 )
    end 
end

initiate()
