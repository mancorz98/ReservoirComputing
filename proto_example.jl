# Source: https://docs.sciml.ai/ReservoirComputing/stable/esn_tutorials/hybrid/#Building-the-Hybrid-Echo-State-Network

using DifferentialEquations

u0 = [1.0, 0.4, 20.0]
tspan = (0.0, 1000.0)
datasize = 100000
tsteps = range(tspan[1], tspan[2]; length = datasize)

function lorenz(du, u, p, t)
	p =  [10.03, 28.23, 2.67] 
	du[1] = p[1] * (u[2] - u[1])
	du[2] = u[1] * (p[2] - u[3]) - u[2]
	du[3] = u[1] * u[2] - p[3] * u[3]
end

ode_prob = ODEProblem(lorenz, u0, tspan)
ode_sol = solve(ode_prob; saveat = tsteps)
ode_data = Array(ode_sol)

train_len = 10000

input_data = ode_data[:, 1:train_len]
target_data = ode_data[:, 2:(train_len+1)]
test_data = ode_data[:, (train_len+1):end][:, 1:1000]

predict_len = size(test_data, 2)
tspan_train = (tspan[1], ode_sol.t[train_len])




using Random

# Deterministic part (drift)
function lorenz_drift!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Stochastic part (diffusion)
function lorenz_diffusion!(du, u, p, t)
    # Simple additive noise - same intensity for all components
    noise_intensity = 0.5  # Adjust this value to control noise level
    du[1] = noise_intensity
    du[2] = noise_intensity
    du[3] = noise_intensity
end

# Alternative: State-dependent (multiplicative) noise
function lorenz_diffusion_multiplicative!(du, u, p, t)
    noise_intensity = 0.1  # Usually smaller for multiplicative noise
    du[1] = noise_intensity * abs(u[1])
    du[2] = noise_intensity * abs(u[2])
    du[3] = noise_intensity * abs(u[3])
end

function sde_prior_model_data_generator(u0, tspan, tsteps; 
                                        model_drift = lorenz_drift!,
                                        model_diffusion = lorenz_diffusion!,
                                        p = [10.0, 28.0, 8/3])
    
    # Create SDE problem
    prob = SDEProblem(model_drift, model_diffusion, u0, tspan, p)
    
    # Solve with explicit dt
    dt = tsteps[2] - tsteps[1]  # Extract timestep from tsteps
    sol = solve(prob, EM(); dt=dt, saveat = tsteps)
    
    # Alternative with adaptive timestep solver (no dt needed)
    # sol = solve(prob, SOSRI(); saveat = tsteps)
    
    return Array(sol)
end

using ReservoirComputing, Random
Random.seed!(42)

p = [12.0, 28.0, 8/3]


km = KnowledgeModel(sde_prior_model_data_generator, u0, tspan_train, train_len)


using ReservoirComputing: NonLinearAlgorithm





res_size = 3000
in_size = 500

hesn = HybridESN(km,
	input_data,
	in_size,
	res_size;
	reservoir = rand_sparse,
	nla_type= NLAT3(),
)


output_layer = train(hesn, target_data, StandardRidge(0.001))

output = hesn(Generative(predict_len), output_layer)


using Plots
lorenz_maxlyap = 0.981  # from literature
predict_ts = tsteps[(train_len + 1):(train_len + predict_len)]
lyap_time = (predict_ts .- predict_ts[1]) * (1 / lorenz_maxlyap)
predict_ts = predict_ts .- predict_ts[1] 


p1 = plot(predict_ts, [test_data[1, :] output[1, :]]; label=["actual" "predicted"],
    ylabel="x(t)", linewidth=2.5, xticks=false, yticks=-15:15:15);
p2 = plot(predict_ts, [test_data[2, :] output[2, :]]; label=["actual" "predicted"],
    ylabel="y(t)", linewidth=2.5, xticks=false, yticks=-20:20:20);
p3 = plot(predict_ts, [test_data[3, :] output[3, :]]; label=["actual" "predicted"],
    ylabel="z(t)", linewidth=2.5, xlabel="max(λ)*t", yticks=10:15:40);

plot(p1, p2, p3; plot_title="Lorenz System Coordinates",
    layout=(3, 1), xtickfontsize=12, ytickfontsize=12, xguidefontsize=15,
    yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)