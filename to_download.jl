using CairoMakie
using Waveforms


function MMS_prime!(du, u, p, t)
	mms_params, V_in = p
	X = u[1]
	G = X / mms_params.Ron + (1 - X) / mms_params.Roff
	V = V_in(t) / (1 + mms_params.Rs * G)
	I = V * G

	du[1] = 1 / mms_params.tau * ((1 / (1 + exp(-mms_params.beta * (V - mms_params.Von)))) * (1 - X) - (1 - 1 / (1 + exp(-mms_params.beta * (V - mms_params.Voff)))) * X)
	# end
	return nothing
end

using Serialization: deserialize

p = deserialize("solution.jls")



Base.@kwdef mutable struct MMSParams
	beta::Float64 = 38.4615
	Ron::Float64 = 5e3
	Roff::Float64 = 1e5
	Von::Float64 = 0.2
	Voff::Float64 = 0.8
	tau::Float64 = 1e-3
	Rs::Float64 = 5.11e3
end



function V_in(t)
	freq = 1e2
	amplitude = 1.5
	return amplitude * squarewave(2π * freq * t)
end

p = MMSParams(), V_in

t = 0.0:1e-6:0.01
u0 = [0.5]


using DifferentialEquations
prob = ODEProblem(MMS_prime!, [0.8], (0, 1), p)
tsteps = 0.0:1e-3:0.1
sol = solve(prob, Tsit5(); saveat = tsteps, maxiters = 10000000)
fig = Figure(resolution = (800, 400))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel =
	"State Variable X", title = "Memristor State Evolution")
lines!(ax, sol.t, sol[1, :], color = :blue)
current_figure()


signal = @. squarewave(2π * 1e4 * t)

fig = Figure(resolution = (800, 400))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel =
	"Amplitude", title = "Square Wave Signal")
lines!(ax, t, signal, color = :blue)
current_figure() |> display



using MLDatasets
using Images
using CairoMakie


train_x, train_y = MNIST(split = :train)[:]
test_x, test_y = MNIST(split = :test)[:]
size(train_x), size(test_x)
