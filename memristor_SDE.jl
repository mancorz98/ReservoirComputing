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

image_data = Gray.(reshape(train_x[:, :, 3], 28, 28)')

image_data = Gray.(train_x[:, :, 3]')
image_data = Float32.(image_data)
train_y[3]  # The label for this image
fig = Figure(resolution = (800, 400))

ax1 = Axis(fig[1, 1], title = "MNIST Training Image")
CairoMakie.heatmap!(ax1, image_data, interpolate = false, colormap = :deep)
current_figure() |> display



function generate_single_pulse(
	pulse_width::Number,
	total_length::Int64,
	fill::Number = 0.5;
	amplitude::Number = 1.0,
	offset::Number = 0.0,
	position::Symbol = :center,
)
	# Input validation
	@assert 0 < pulse_width ≤ 1 "pulse_width must be in (0, 1]"
	@assert 0 < fill ≤ 1 "fill must be in (0, 1]"
	@assert total_length > 0 "total_length must be positive"

	# Initialize signal
	signal = fill!(Vector{Float32}(undef, total_length), Float32(offset))

	# Calculate actual pulse length considering fill factor
	active_length = Int(round(pulse_width * total_length))
	pulse_length = Int(round(active_length * fill))

	# Determine pulse position
	start_index = if position == :center
		Int(round((total_length - pulse_length) / 2)) + 1
	elseif position == :left
		1
	elseif position == :right
		total_length - pulse_length + 1
	elseif position isa Number
		# Custom position as fraction of total_length
		max(1, min(total_length - pulse_length + 1, Int(round(position * total_length))))
	else
		throw(ArgumentError("position must be :center, :left, :right, or a number in [0,1]"))
	end

	# Ensure indices are within bounds
	end_index = min(start_index + pulse_length - 1, total_length)
	start_index = max(1, start_index)

	# Set pulse amplitude
	signal[start_index:end_index] .= Float32(amplitude + offset)

	return signal
end


function heaviside(x)
	0.5 * (sign(x) + 1)
end



function generate_carrier_wave(
	pulse_width::Number,
	pulses::Int64,
	fill::Number = 0.5;
	fs::Number = 1e4,
	amplitude::Number = 1.0,
	offset::Number = 0.0,
	gap_ratio::Number = 0.0,
	pulse_variation::Function = (i, n) -> 1.0,  # Fixed: default function now accepts 2 args
	add_noise::Number = 0.0,
)
	# Input validation
	@assert pulses > 0 "Number of pulses must be positive"
	@assert 0 ≤ gap_ratio < 1 "gap_ratio must be in [0, 1)"
	@assert fs > 0 "Sampling frequency must be positive"
	@assert add_noise ≥ 0 "Noise level must be non-negative"

	# Calculate period length including gap
	period_length = Int(round(fs * (1 + gap_ratio)))
	pulse_period = Int(round(fs))

	# Pre-allocate output array for efficiency
	total_length = period_length * pulses
	signal = Vector{Float32}(undef, total_length)

	# Generate pulses with potential variations
	for i in 1:pulses
		# Apply pulse variation function (e.g., for amplitude modulation)
		current_amplitude = amplitude * 1.0

		# Generate single pulse
		single_pulse = generate_single_pulse(
			pulse_width,
			pulse_period,
			fill;
			amplitude = current_amplitude,
			offset = offset,
		)

		# Calculate indices for this pulse
		start_idx = (i - 1) * period_length + 1
		end_idx = start_idx + pulse_period - 1

		# Place pulse in signal
		signal[start_idx:end_idx] = single_pulse

		# Fill gap with offset if gap exists
		if gap_ratio > 0 && i < pulses
			gap_start = end_idx + 1
			gap_end = min(i * period_length, total_length)
			signal[gap_start:gap_end] .= Float32(offset)
		end
	end

	# Add noise if requested
	if add_noise > 0
		signal .+= Float32(add_noise) .* randn(Float32, length(signal))
	end

	return signal
end


function generate_pulse(
	duty_cycle::Number = 0.5,  # Fraction of period that is "on"
	period::Number = 1.0;       # Total period length
	fs::Number = 1e3,           # Sampling frequency
)
	dt = 1 / fs
	# Create time array for one period
	t = 0:dt:(period-dt)

	# Generate rectangular pulse
	pulse_width = duty_cycle * period
	y = [ti < pulse_width ? 1.0 : 0.0 for ti in t]

	return t, y
end



function generate_carrier_wave(
	period::Number,
	amps::AbstractVector{<:Number};
	duty_cycle::Number = 0.5,
	fs::Number = 1e3,
)
	n_periods = length(amps)
	dt = 1 / fs
	total_time = period * n_periods

	# Pre-allocate arrays for efficiency
	t = 0:dt:(total_time-dt)
	signal = zeros(Float64, length(t))

	# Generate periodic rectangular wave
	pulse_width = duty_cycle * period
	for (i, ti) in enumerate(t)
		# Check if we're in the "on" portion of current period
		phase = mod(ti, period)
		signal[i] = (phase < pulse_width ? 1.0 : 0.0) * amps[floor(Int, ti / period)+1]
		# @show floor(ti / period)  # Debug: show current period index
	end

	return collect(t), signal
end


amps = [1.0, 0.5, 0.8, 1.2, 0.3]
(times, signal) = generate_carrier_wave(1, amps)
size(times), size(signal)

lines(times, signal, color = :blue) |> display



function generate_random_memristor(
	n_memristors::Int64,
)::Vector{MMSParams}
	ref_memristor = MMSParams(tau = 1e-4)
	memristors = Vector{MMSParams}(undef, n_memristors)
	for i in 1:n_memristors
		memristors[i] = MMSParams(
			Ron = ref_memristor.Ron * (0.8 + 0.4 * rand()),
			Roff = ref_memristor.Roff * (0.8 + 0.4 * rand()),
			Von = ref_memristor.Von * (0.8 + 0.4 * rand()),
			Voff = -ref_memristor.Voff * (0.8 + 0.4 * rand()),
			tau = ref_memristor.tau * (0.8 + 0.4 * rand()),
		)
	end
	return memristors
end

using DataInterpolations: AkimaInterpolation, ExtrapolationType
using ProgressMeter
memristors = generate_random_memristor(train_x |> size |> first)



function extract_patches(image, kernel_size = (3, 3), stride = 2)
	patches = []
	h, w = size(image)
	kh, kw = kernel_size

	for i in 1:stride:(h-kh+1)
		for j in 1:stride:(w-kw+1)
			patch = image[i:i+kh-1, j:j+kw-1]
			push!(patches, vec(patch))
		end
	end
	return patches
end


extract_patches(train_x[:, :, 1]', (5, 5), 3) |> length


x_vals = Array{Float32}(undef, train_x |> size |> first, size(train_x, 3))
@showprogress 1 "ComputingImageData" for j in 1:size(train_x, 3)
	image_data = train_x[:, :, j]'
	# image_data = train_x[:, :, 2]'
	label = train_y[j]  # The label for this image
	Gray.(image_data)

	for i in 1:size(image_data, 1)
		(times, signal) = generate_carrier_wave(1e-4, image_data[i, :] .* 1; duty_cycle = 0.8, fs = 1e7)
		V_in_f = t -> interp1(times, signal, t, left = 0.0, right = 0.0)

		V_in_f = AkimaInterpolation(signal, times; extrapolation = ExtrapolationType.Extension)
		p = (memristors[i], V_in_f)
		prob = ODEProblem(MMS_prime!, [0.00], (0.0, maximum(times)), p)
		sol = solve(prob, Tsit5(); saveat = [times[end]],
			maxiters = Int(1e4))
		x_vals[i, j] = sol[end][1]

		# fig = Figure(size = (800, 400))
		# ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Memristor State X", title = "Memristor State Evolution for Row $i of Image $label")
		# lines!(ax, sol, color = :blue)
		# ylims!(ax, 0, 1)

		# ax2 = Axis(fig[1, 2], xlabel = "Time (s)", ylabel = "Input Voltage (V)", title = "Input Voltage Signal for Row $i of Image $label")
		# lines!(ax2, times, signal, color = :red)
		# current_figure() |> display
		# x_vals[i] = sol[end][1]
		# sleep(0.1)
	end
end




x_vals = Array{Float32}(undef, train_x |> size |> first, size(train_x, 3))

kernel_size = 5
stride = 3
n_patches_h = div(28 - kernel_size, stride) + 1  # = 8
n_patches_w = div(28 - kernel_size, stride) + 1  # = 8
n_patches = n_patches_h * n_patches_w  # = 64
x_vals = Array{Float32}(undef, n_patches + 1, size(train_x, 3))

@showprogress 1 "ComputingImageData" for j in 1:size(train_x, 3)
	image_data = train_x[:, :, j]'
	patches = extract_patches(image_data, (5, 5), 3)

	# image_data = train_x[:, :, 2]'
	label = train_y[j]  # The label for this image
	Gray.(image_data)

	for (i, patch) in enumerate(patches)
		(times, signal) = generate_carrier_wave(1e-5, patch .* 1; duty_cycle = 0.8, fs = 1e7)
		V_in_f = t -> interp1(times, signal, t, left = 0.0, right = 0.0)

		V_in_f = AkimaInterpolation(signal, times; extrapolation = ExtrapolationType.Extension)
		p = (memristors[i%length(memristors)+1], V_in_f)
		prob = ODEProblem(MMS_prime!, [0.00], (0.0, maximum(times)), p)
		sol = solve(prob, Tsit5(); saveat = times,
			maxiters = Int(1e4))
		x_vals[i, j] = sol[end][1]

		# fig = Figure(size = (800, 400))
		# ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Memristor State X", title = "Memristor State Evolution for Row $i of Image $label")
		# lines!(ax, sol, color = :blue)
		# ylims!(ax, 0, 1)

		# ax2 = Axis(fig[1, 2], xlabel = "Time (s)", ylabel = "Input Voltage (V)", title = "Input Voltage Signal for Row $i of Image $label")
		# lines!(ax2, times, signal, color = :red)
		# current_figure() |> display
		# x_vals[i] = sol[end][1]
		# sleep(0.1)
	end
	# Add bias term
	x_vals[end, j] = 1.0
end






size(x_vals)

y_labels = train_y
y_labels = one_hot_encoding(y_labels, 10)'


x_vals_t = x_vals'



train_y[2]
y_labels


temp_matrix = rand(Float32, size(x_vals_t, 2), size(y_labels, 2))


x_vals_t * temp_matrix |> size


H = x_vals_t
Y = y_labels
@show size(H)
@show size(Y)




function one_hot_encoding(labels::Vector{Int}, num_classes::Int)
	one_hot = zeros(Float32, num_classes, length(labels))
	for (i, label) in enumerate(labels)
		one_hot[label+1, i] = 1.0
	end
	return one_hot
end

using LinearAlgebra

function efficient_ridge_regression(H, Y, λ = 0.001)
	# Work in feature space
	H_T_H = H' * H           # (28, 28) 
	H_T_Y = H' * Y           # (28, 10)

	# Add regularization and ensure symmetry
	H_T_H_reg = Hermitian(H_T_H + λ * I)

	# Solve using Cholesky (fastest for positive definite)
	W_out = cholesky(H_T_H_reg) \ H_T_Y

	return W_out  # (28, 10)
end


function qr_ridge_regression(H, Y, λ = 0.001)
	m, n = size(H)

	# QR decomposition of H
	Q, R = qr(H)

	# Solve the regularized system
	W_out = (R' * R + λ * I) \ (R' * Q' * Y)

	return W_out  # (28, 10)
end


W_out = qr_ridge_regression(H, Y, 0.00001)
W_out = efficient_ridge_regression(H, Y, 0.0)
y_pred = H * W_out
y_pred_labels = map(i -> findmax(y_pred[i, :])[2] - 1, 1:size(y_pred, 1))

accuracy = sum(y_pred_labels .== train_y) / length(train_y)


println("Training Accuracy: ", accuracy * 100, "%")



unpred_ids = findall(y_pred_labels .!= train_y)

rand_ind = unpred_ids[rand(1:length(unpred_ids))]
image_data = train_x[:, :, rand_ind]'
image_data = Gray.(image_data)

fig = Figure(size = (400, 400))
ax1 = Axis(fig[1, 1], title = "MNIST Training Image (Misclassified) - True: $(train_y[rand_ind]), Pred: $(y_pred_labels[rand_ind])",
	xaxisposition = :top,      # x-axis on top instead of bottom
	yaxisposition = :right,
	yreversed = true)
CairoMakie.image!(ax1, image_data', interpolate = false, colormap = :deep)
current_figure() |> display











image(image_data, colormap = :deep) |> display













times
lines(times, signal, color = :blue) |> display

lines(generate_carrier_wave(0.1, 5)...)

using Statistics
times[2:end] - times[1:end-1] |> mean


fig = Figure(; size = (800, 400))
ax = Axis(fig[1, 1], xlabel = "Sample Index", ylabel = "Amplitude", title = "Generated Carrier Wave with Pulses")
lines!(ax, generate_carrier_wave(0.1, 5, 1.0; fs = 1000, amplitude = 1.0, offset = 0.0, gap_ratio = 0.5, add_noise = 0.0), color = :blue)

current_figure() |> display

for row in 1:4
end






