using LinearAlgebra
using Random
using DifferentialEquations
using CairoMakie
using Statistics

"""
Echo State Network with Lorenz system dynamics in the reservoir.

The reservoir states evolve according to modified Lorenz equations,
providing rich chaotic dynamics for temporal pattern learning.

parameters:
- input_size: Dimension of input vector
- reservoir_size: Total number of reservoir neurons (should be multiple of 3)
- output_size: Dimension of output vector
- leaking_rate: Rate at which reservoir state updates
- lorenz_coupling: Coupling strength between Lorenz systems in the reservoir
- σ, ρ, β: Standard Lorenz system parameters
"""
mutable struct LorenzESN
	input_size::Int
	reservoir_size::Int
	output_size::Int
	leaking_rate::Float64
	lorenz_coupling::Float64

	# Lorenz parameters
	σ::Float64  # sigma
	ρ::Float64  # rho
	β::Float64  # beta

	# Network weights
	W_in::Matrix{Float64}
	W_res::Matrix{Float64}
	W_out::Matrix{Float64}
	bias_out::Vector{Float64}

	# State
	reservoir_state::Vector{Float64}

	# Derived parameters
	num_lorenz_systems::Int
	actual_reservoir_size::Int

	function LorenzESN(input_size::Int, reservoir_size::Int, output_size::Int;
		spectral_radius::Float64 = 0.9,
		input_scaling::Float64 = 1.0,
		leaking_rate::Float64 = 1.0,
		lorenz_coupling::Float64 = 0.1,
		σ::Float64 = 11.0,
		ρ::Float64 = 28.0,
		β::Float64 = 8.0 / 3.0)

		# Ensure reservoir size is multiple of 3 for Lorenz triplets
		num_lorenz_systems = reservoir_size ÷ 3
		actual_reservoir_size = num_lorenz_systems * 3

		# Initialize input weights
		W_in = randn(actual_reservoir_size, input_size) * input_scaling

		# Initialize reservoir coupling weights (sparse connectivity between Lorenz systems)
		W_res = make_sparse(randn(actual_reservoir_size, actual_reservoir_size), 0.1)

		# Scale to desired spectral radius
		eigenvals = eigvals(W_res)
		current_spectral_radius = maximum(abs.(eigenvals))
		W_res = W_res * (spectral_radius / current_spectral_radius)

		# Initialize output weights
		W_out = randn(output_size, actual_reservoir_size) * 0.1
		bias_out = zeros(output_size)

		# Initialize reservoir state
		reservoir_state = zeros(actual_reservoir_size)

		new(input_size, reservoir_size, output_size, leaking_rate, lorenz_coupling,
			σ, ρ, β, W_in, W_res, W_out, bias_out, reservoir_state,
			num_lorenz_systems, actual_reservoir_size)
	end
end

"""Make matrix sparse by randomly setting elements to zero"""
function make_sparse(matrix::Matrix{Float64}, sparsity::Float64 = 0.1)
	mask = rand(size(matrix)...) .< sparsity
	return matrix .* mask
end

"""Compute Lorenz derivatives for the entire reservoir state"""
function lorenz_derivatives(esn::LorenzESN, state::Vector{Float64})
	# Reshape to (num_systems, 3) for easier processing
	lorenz_state = reshape(state, 3, esn.num_lorenz_systems)'

	# Extract x, y, z components
	x = lorenz_state[:, 1]
	y = lorenz_state[:, 2]
	z = lorenz_state[:, 3]

	# Compute Lorenz derivatives for each system
	dx_dt = esn.σ .* (y .- x)
	dy_dt = x .* (esn.ρ .- z) .- y
	dz_dt = x .* y .- esn.β .* z

	# Stack derivatives and flatten
	derivatives = hcat(dx_dt, dy_dt, dz_dt)'
	return vec(derivatives)
end

"""Apply Lorenz dynamics to reservoir state using RK4 integration"""
function lorenz_dynamics(esn::LorenzESN, state::Vector{Float64}, dt::Float64 = 0.001)
	k1 = dt * lorenz_derivatives(esn, state)
	k2 = dt * lorenz_derivatives(esn, state + 0.5 * k1)
	k3 = dt * lorenz_derivatives(esn, state + 0.5 * k2)
	k4 = dt * lorenz_derivatives(esn, state + k3)

	new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
	return new_state
end


"""Forward pass through the ESN"""
function forward(esn::LorenzESN, input_sequence::Array{Float64})
	if ndims(input_sequence) == 2
		seq_len, input_dim = size(input_sequence)
		batch_size = 1
		input_sequence = reshape(input_sequence, seq_len, 1, input_dim)
	else
		seq_len, batch_size, input_dim = size(input_sequence)
	end

	# Initialize states for batch
	reservoir_states = repeat(esn.reservoir_state', batch_size, 1)
	all_states = Array{Float64}(undef, seq_len, batch_size, esn.actual_reservoir_size)

	for t in 1:seq_len
		current_input = input_sequence[t, :, :]  # (batch_size, input_size)

		for b in 1:batch_size
			# Apply Lorenz dynamics to evolve the previous state
			lorenz_evolved_state = lorenz_dynamics(esn, reservoir_states[b, :])

			# Standard ESN update following the article's equation:
			# ht = f((1 - k) * ut * Win + k * ht-1 * Wh + b)
			# where f() is tanh, k is leak rate

			# Input contribution: ut * Win (note: input from left)
			input_contribution = (current_input[b, :]' * esn.W_in')'

			# Reservoir contribution: ht-1 * Wh (using Lorenz-evolved state)  
			reservoir_contribution = (lorenz_evolved_state' * esn.W_res')'

			# Apply correct leaking rate formulation from the article
			linear_combination = (1 - esn.leaking_rate) * input_contribution +
								 esn.leaking_rate * reservoir_contribution

			# Apply activation function to the entire linear combination
			new_state = tanh.(linear_combination)

			reservoir_states[b, :] = new_state
		end

		all_states[t, :, :] = reservoir_states
	end

	# Compute outputs
	outputs = Array{Float64}(undef, seq_len, batch_size, esn.output_size)
	for t in 1:seq_len
		for b in 1:batch_size
			outputs[t, b, :] = esn.W_out * all_states[t, b, :] + esn.bias_out
		end
	end

	if batch_size == 1
		return dropdims(outputs, dims = 2), dropdims(all_states, dims = 2)
	else
		return outputs, all_states
	end
end


"""Reset reservoir state"""
function reset_state!(esn::LorenzESN)
	esn.reservoir_state .= 0.0
end

"""Generate Lorenz attractor data for testing"""
function generate_lorenz_data(num_steps::Int = 1000, dt::Float64 = 0.01;
	σ::Float64 = 10.0, ρ::Float64 = 28.0, β::Float64 = 8.0 / 3.0)

	function lorenz!(du, u, p, t)
		σ, ρ, β = p
		du[1] = σ * (u[2] - u[1])
		du[2] = u[1] * (ρ - u[3]) - u[2]
		du[3] = u[1] * u[2] - β * u[3]
	end

	u0 = [-10, -10, 25]
	tspan = (0.0, num_steps * dt)
	p = [σ, ρ, β]

	prob = ODEProblem(lorenz!, u0, tspan, p)
	sol = solve(prob, Tsit5(), saveat = dt)

	return Matrix(sol)'  # Shape: (num_steps, 3)
end

"""Train ESN using ridge regression for output weights"""
function train_esn!(esn::LorenzESN, X_train::Array{Float64, 3}, y_train::Array{Float64, 3};
	ridge_param::Float64 = 1e-6)

	println("Training ESN with Lorenz reservoir dynamics...")

	# Collect reservoir states for all training sequences
	all_reservoir_states = Vector{Matrix{Float64}}()
	all_targets = Vector{Matrix{Float64}}()

	num_sequences = size(X_train, 1)

	for i in 1:num_sequences
		reset_state!(esn)

		# Manual forward pass to collect reservoir states
		reservoir_states = Array{Float64}(undef, size(X_train, 2), esn.actual_reservoir_size)

		for t in 1:size(X_train, 2)
			current_input = X_train[i, t, :]

			# Apply Lorenz dynamics to evolve the previous state
			lorenz_evolved_state = lorenz_dynamics(esn, esn.reservoir_state)

			# Standard ESN update following the article's equation
			input_contribution = (current_input' * esn.W_in')'
			reservoir_contribution = (lorenz_evolved_state' * esn.W_res')'

			# Apply correct leaking rate formulation
			linear_combination = (1 - esn.leaking_rate) * input_contribution +
								 esn.leaking_rate * reservoir_contribution

			# Apply activation function to the entire linear combination
			new_state = tanh.(linear_combination)

			esn.reservoir_state = new_state
			reservoir_states[t, :] = esn.reservoir_state
		end

		push!(all_reservoir_states, reservoir_states)
		push!(all_targets, y_train[i, :, :])
	end

	# Concatenate all data
	X_reservoir = vcat(all_reservoir_states...)  # (total_timesteps, reservoir_size)
	y_flat = vcat(all_targets...)               # (total_timesteps, output_size)

	# Ridge regression for output weights
	I_reg = Matrix{Float64}(I, size(X_reservoir, 2), size(X_reservoir, 2))

	# Solve: W_out = (X^T X + λI)^{-1} X^T y
	XTX = X_reservoir' * X_reservoir
	XTy = X_reservoir' * y_flat
	W_out_optimal = (XTX + ridge_param * I_reg) \ XTy

	# Set the optimal weights
	esn.W_out = W_out_optimal'
	esn.bias_out .= 0.0

	println("Training completed!")
end

"""Example training and testing script"""
function train_esn_example()
	# Generate synthetic data (predicting next step of Lorenz system)
	println("Generating Lorenz data...")


	data = generate_lorenz_data(4000, 0.01)

	# Prepare sequences
	seq_length = 2000
	X = Array{Float64}(undef, size(data, 1) - seq_length, seq_length, 3)
	y = Array{Float64}(undef, size(data, 1) - seq_length, seq_length, 3)

	for i in 1:(size(data, 1)-seq_length)
		X[i, :, :] = data[i:(i+seq_length-1), :]
		y[i, :, :] = data[(i+1):(i+seq_length), :]
	end

	# Split data
	train_size = Int(floor(0.8 * size(X, 1)))
	X_train, X_test = X[1:train_size, :, :], X[(train_size+1):end, :, :]
	y_train, y_test = y[1:train_size, :, :], y[(train_size+1):end, :, :]

	# Create ESN
	esn = LorenzESN(3, 600,  # 100 Lorenz systems  
		3;
		spectral_radius = 0.95,
		input_scaling = 1.0,
		leaking_rate = 0.2,
		lorenz_coupling = 0.05)

	# Train the model
	train_esn!(esn, X_train, y_train)

	# Test the model
	test_predictions = Vector{Matrix{Float64}}()
	test_targets = Vector{Matrix{Float64}}()

	num_test_sequences = min(5, size(X_test, 1))

	for i in 1:num_test_sequences
		reset_state!(esn)
		pred, _ = forward(esn, X_test[i, :, :])
		push!(test_predictions, pred)
		push!(test_targets, y_test[i, :, :])
	end

	# Calculate MSE
	mse = mean([sum((pred - target) .^ 2) for (pred, target) in zip(test_predictions, test_targets)])
	println("Test MSE: ", mse)

	return esn, test_predictions, test_targets, data
end

"""Plot results using CairoMakie"""
function plot_results(esn::LorenzESN, predictions, targets, original_data)

	# Set up the figure with subplots
	fig = Figure(resolution = (1200, 800))

	# Plot original Lorenz attractor
	ax1 = Axis(fig[1, 1], title = "Original Lorenz Attractor (X-Z plane)",
		xlabel = "X", ylabel = "Z")
	lines!(ax1, original_data[1:1000, 1], original_data[1:1000, 3],
		color = :blue, linewidth = 1.5)

	if !isempty(predictions)
		pred = predictions[1]
		target = targets[1]

		# Plot X component prediction
		ax2 = Axis(fig[1, 2], title = "X Component Prediction")
		lines!(ax2, 1:length(target[:, 1]), target[:, 1],
			label = "Target X", color = :blue, linewidth = 2)
		lines!(ax2, 1:length(pred[:, 1]), pred[:, 1],
			label = "Predicted X", color = :red, linewidth = 2, linestyle = :dash)
		axislegend(ax2)

		# Plot Y component prediction  
		ax3 = Axis(fig[1, 3], title = "Y Component Prediction")
		lines!(ax3, 1:length(target[:, 2]), target[:, 2],
			label = "Target Y", color = :blue, linewidth = 2)
		lines!(ax3, 1:length(pred[:, 2]), pred[:, 2],
			label = "Predicted Y", color = :red, linewidth = 2, linestyle = :dash)
		axislegend(ax3)

		# Plot Z component prediction
		ax4 = Axis(fig[2, 1], title = "Z Component Prediction")
		lines!(ax4, 1:length(target[:, 3]), target[:, 3],
			label = "Target Z", color = :blue, linewidth = 2)
		lines!(ax4, 1:length(pred[:, 3]), pred[:, 3],
			label = "Predicted Z", color = :red, linewidth = 2, linestyle = :dash)
		axislegend(ax4)

		# Phase space comparison
		ax5 = Axis(fig[2, 2], title = "Phase Space (X-Z)",
			xlabel = "X", ylabel = "Z")
		lines!(ax5, target[:, 1], target[:, 3],
			label = "Target", color = :blue, linewidth = 2)
		lines!(ax5, pred[:, 1], pred[:, 3],
			label = "Predicted", color = :red, linewidth = 2, linestyle = :dash)
		axislegend(ax5)

		# Error plot
		ax6 = Axis(fig[2, 3], title = "Absolute Error")
		error = abs.(pred - target)
		lines!(ax6, 1:length(error[:, 1]), error[:, 1],
			label = "X error", color = :red, linewidth = 2)
		lines!(ax6, 1:length(error[:, 2]), error[:, 2],
			label = "Y error", color = :green, linewidth = 2)
		lines!(ax6, 1:length(error[:, 3]), error[:, 3],
			label = "Z error", color = :purple, linewidth = 2)
		axislegend(ax6)
	end

	return fig
end

# Main execution
function main()
	esn, predictions, targets, original_data = train_esn_example()
	fig = plot_results(esn, predictions, targets, original_data)
	display(fig)
	return fig
end

# Uncomment to run
main()
