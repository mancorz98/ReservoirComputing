using DifferentialEquations
using Serialization: deserialize 
using CairoMakie
using Waveforms
using MLDatasets
using Images
using CairoMakie
using Random
using Statistics
using DataInterpolations: AkimaInterpolation, ExtrapolationType
using ProgressMeter
using LinearAlgebra

using CairoMakie: Axis

train_x, train_y = MNIST(split = :train)[:]
test_x, test_y = MNIST(split = :test)[:]
# train_indices = randperm(size(train_x, 3))#[1:1000]
# test_indices = randperm(size(test_x, 3))#[1:200]
train_x = train_x[:, :, train_indices]
train_y = train_y[train_indices]
test_x = test_x[:, :, test_indices]
test_y = test_y[test_indices]

size(train_x), size(test_x)

image_data = Gray.(reshape(train_x[:, :, 10], 28, 28)')

image_data = Gray.(train_x[:, :, 10]')
image_data = Float32.(image_data)
train_y[10]  # The label for this image
fig = Figure(resolution = (800, 800))

ax1 = Axis(fig[1, 1],
    xaxisposition = :top,      # x-axis on top instead of bottom
    yaxisposition = :right,
    yreversed = true,
    xticksvisible = false,
    xticklabelsvisible = false,
    yticksvisible = false,
    yticklabelsvisible = false
    )
CairoMakie.heatmap!(ax1, transpose(image_data), interpolate = false, colormap = :grays)
current_figure() |> display
save("mnist_sample_image.pdf", fig)


fig = Figure(resolution = (800, 800))

ax1 = Axis(fig[1, 1],
    xaxisposition = :top,      # x-axis on top instead of bottom
    yaxisposition = :right,
    yreversed = true,
    xticksvisible = false,
    xticklabelsvisible = false,
    yticksvisible = false,
    yticklabelsvisible = false
    )
CairoMakie.heatmap!(ax1, transpose(image_data), interpolate = false, colormap = :grays)
current_figure() |> display
save("mnist_sample_image.pdf", fig)



patch = transpose(image_data)[11:18, 8:15]  # 8x8
patch


fig = Figure(resolution = (800, 800))

ax1 = Axis(fig[1, 1],
    xaxisposition = :top,      # x-axis on top instead of bottom
    yaxisposition = :right,
    yreversed = true,
    xticksvisible = false,
    xticklabelsvisible = false,
    yticksvisible = false,
    yticklabelsvisible = false
    )

CairoMakie.heatmap!(ax1, patch, interpolate = false, colormap = :grays)
current_figure() |> display
save("mnist_sample_patch.pdf", fig)





for row in 1:size(patch, 1)
    # Get pixel values from this row
    row_pixels = transpose(patch)[row, :]  # 8 pixels in this row
    
    # Generate carrier wave: 8 pulses, amplitudes = pixel values
    times, v_signal = generate_carrier_wave(
        1e-5, 
        row_pixels .* 1.5;
        duty_cycle = 0.8, 
        fs = 1e7
    )

    fig = Figure(resolution = (800, 200))
    ax = Axis(fig[1, 1], ylabel = "Voltage (V)", 
                xticklabelsvisible = false,
                xtrimspine = true,
                xgridvisible = false,
                xticksvisible = false
                )
    lines!(ax, times, v_signal, color = :blue)
    hidespines!(ax)
    current_figure() |> display
    save("mnist_patch_row_$(row)_carrier_wave.pdf", fig)
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



function extract_patches(image, kernel_size = (3, 3), stride = 2)
	patches = Array{Float32, 3}(undef, 0,  kernel_size[1], kernel_size[2])
	h, w = size(image)
	kh, kw = kernel_size

	for i in 1:stride:(h-kh+1)
		for j in 1:stride:(w-kw+1)
			patch = image[i:i+kh-1, j:j+kw-1]
            patches = cat(patches, reshape(Float32.(patch), 1, kh, kw); dims = 1)
		end
	end
	return patches
end











p = deserialize("solution.jls")


Base.@kwdef mutable struct MMSParams
	beta::Float64 = 38.4615
	Ron::Float64 = 5e3
	Roff::Float64 = 1e5
	Von::Float64 = 0.2
	Voff::Float64 = -0.1
	tau::Float64 = 1e-3
	Rs::Float64 = 5.11e3
end




function crossbar_8x8!(du, u, p, t)
    """
    Crossbar array dynamics: 8×8 = 64 memristors
    
    Direct calculation approach:
    1. Compute all conductances G[i,j]
    2. Calculate voltage across each memristor considering column loading
    3. Update memristor states dX/dt
    
    u: state vector [X₁, X₂, ..., X₆₄] - all memristor states
    p: (memristor_params_matrix, V_rows_function, R_series)
    
    Crossbar physics:
    - V_row[i] applied to row i
    - Each column has total conductance G_col[j] = Σᵢ G[i,j]
    - Column voltage: V_col[j] = (Σᵢ V_row[i]×G[i,j]) / (G_col[j] + G_series[j])
    - Memristor voltage: V_mem[i,j] = V_row[i] - V_col[j]
    """
    
    memristor_params, V_rows_func, R_series = p
    
    # Reshape state vector to 8×8 matrix
    X_matrix = reshape(u, 8, 8)
    
    # Get current row voltages
    V_rows = V_rows_func(t)  # [V₁, V₂, ..., V₈]
    
    # Step 1: Calculate all memristor conductances (EACH MEMRISTOR HAS DIFFERENT Ron, Roff)
    G_matrix = zeros(Float64, 8, 8)
    for i in 1:8, j in 1:8
        X = X_matrix[i, j]
        params = memristor_params[i, j]  # Each memristor has its own parameters!
        G_matrix[i, j] = X / params.Ron + (1 - X) / params.Roff
    end
    
    # Step 2: Calculate column voltages
    # Each column acts as a current divider with series resistance
    G_series = 1.0 ./ R_series  # Convert to conductance (each column can have different R_series)
    
    V_cols = zeros(Float64, 8)
    for j in 1:8
        # Total conductance in column j
        G_col_total = sum(G_matrix[:, j])
        
        # Weighted sum of row voltages (current contribution from each row)
        I_col_numerator = sum(V_rows[i] * G_matrix[i, j] for i in 1:8)
        
        # Column voltage using current divider
        V_cols[j] = I_col_numerator / (G_col_total + G_series[j])
    end
    
    # Step 3: Calculate voltage across each memristor and update states
    du_matrix = zeros(Float64, 8, 8)
    
    for i in 1:8, j in 1:8
        X = X_matrix[i, j]
        params = memristor_params[i, j]  # Use individual memristor parameters
        
        # Voltage across memristor (i,j)
        V_mem = V_rows[i] - V_cols[j]
        
        # MMS model dynamics (using individual memristor's Von, Voff, tau, beta)
        f_on = 1 / (1 + exp(-params.beta * (V_mem - params.Von)))
        f_off = 1 / (1 + exp(-params.beta * (V_mem - params.Voff)))
        
        du_matrix[i, j] = (1 / params.tau) * (f_on * (1 - X) - (1 - f_off) * X)
    end
    
    # Flatten back to vector form
    du[:] = vec(du_matrix)
    
    return nothing
end

function V_in(t)
	freq = 1e3
	amplitude = 1.5
	return [amplitude * sin(2π * freq * t) for i in 1:8 ]
end



function generate_random_memristor(
	n_memristors::Int64,
    range_factor::Float64 = 0.25
)::Array{MMSParams}
	ref_memristor = MMSParams(tau = 1e-4)
	memristors = Array{MMSParams}(undef, n_memristors, n_memristors)
	for i in 1:n_memristors
        for j in 1:n_memristors
            memristors[i, j] = MMSParams(
                Ron = ref_memristor.Ron * (1 - range_factor + 2 * range_factor * rand()),
                Roff = ref_memristor.Roff * (1 - range_factor + 2 * range_factor * rand()),
                Von = abs(ref_memristor.Von * (1 - range_factor + 2 * range_factor * rand())),
                Voff = abs(ref_memristor.Voff * (1 - range_factor + 2 * range_factor * rand())) * -1., 
                tau = ref_memristor.tau * (1 - range_factor + 2 * range_factor * rand()),
            )
        end
	end
	return memristors
end

p = generate_random_memristor(8, 0.5)

size(p)
p = p, V_in, fill(5.11e3, 8)  # Series resistances for each column

t = 0.0:1e-6:0.01
u0 = fill(0.5f0, 64)  # Initial states for 64 memristors





prob = ODEProblem(crossbar_8x8!, u0, (0.0, 0.01), p)
sol = solve(prob, Tsit5(), saveat = t, reltol = 1e-4, abstol = 1e-6, maxiters = 1e12)    






kernel_size = 8
stride = 3
n_patches_h = div(size(train_x, 1) - kernel_size, stride) + 1  # = 8
n_patches_w = div(size(train_x, 2) - kernel_size, stride) + 1  # = 8
n_patches = n_patches_h * n_patches_w  # = 64
x_vals = Array{Float32}(undef, n_patches + 1, size(train_x, 3))

memristors = generate_random_memristor(8, 0.5)




# Calculate output currents after simulation
function calculate_column_currents(X_matrix::Matrix{Float64}, 
                                   V_rows::Vector{Float64},
                                   memristor_params::Matrix{MMSParams},
                                   R_series::Vector{Float64})
    """
    Calculate column currents (reservoir outputs) after simulation
    Each memristor uses its own Ron, Roff parameters
    """
    
    # Calculate conductances (each memristor has different Ron, Roff)
    G_matrix = zeros(Float64, 8, 8)
    for i in 1:8, j in 1:8
        X = X_matrix[i, j]
        params = memristor_params[i, j]
        G_matrix[i, j] = X / params.Ron + (1 - X) / params.Roff
    end
    
    # Calculate column voltages
    G_series = 1.0 ./ R_series
    V_cols = zeros(Float64, 8)
    
    for j in 1:8
        G_col_total = sum(G_matrix[:, j])
        I_col_numerator = sum(V_rows[i] * G_matrix[i, j] for i in 1:8)
        V_cols[j] = I_col_numerator / (G_col_total + G_series[j])
    end
    
    # Column currents through series resistances
    I_cols = V_cols ./ R_series
    
    return I_cols, V_cols
end


R_series = fill(5.11e3, 8)  # Series resistances for each column


kernel_size = 8
stride = 3
n_patches_per_dim = div(28 - kernel_size, stride) + 1  # 7
n_patches = n_patches_per_dim * n_patches_per_dim  # 49

# CORRECT dimension calculation
n_features_per_patch = kernel_size  # 8 column currents per patch
n_total_features = n_patches * n_features_per_patch + 1  # 49*8 + 1 = 393

# Create feature matrices with CORRECT size
x_vals = zeros(Float32, n_total_features, size(train_x, 3))  # (393, 60000)
test_vals = zeros(Float32, n_total_features, size(test_x, 3))  # (393, 10000)

println("Feature matrix dimensions: ", size(x_vals))  # Should print (393, 60000)
println("Number of patches per image: ", n_patches)   # Should print 49
println("Features per patch: ", n_features_per_patch) # Should print 8
println("Total features: ", n_total_features)         # Should print 393


# Training data processing
@showprogress 1 "Processing training images" for j in 1:size(train_x, 3)
    image_data = train_x[:, :, j]'
    patches = extract_patches(image_data, (kernel_size, kernel_size), stride)  # (49, 8, 8)
    
    for i in 1:size(patches, 1)  # Iterate over all patches
        # Get the i-th patch as 8×8 matrix
        patch = patches[i, :, :]  # Extract 8×8 from 3D array
        
        # Create voltage functions for each ROW
        v_funcs = Vector{AkimaInterpolation}(undef, kernel_size)
        times = nothing
        
        for row in 1:kernel_size
            # Get pixel values from this row
            row_pixels = patch[row, :]  # 8 pixels in this row
            
            # Generate carrier wave: 8 pulses, amplitudes = pixel values
            times, v_signal = generate_carrier_wave(
                1e-5, 
                row_pixels .* 1.5;
                duty_cycle = 0.8, 
                fs = 1e7
            )
            
            # Create interpolation for this row
            v_funcs[row] = AkimaInterpolation(
                v_signal, 
                times; 
                extrapolation = ExtrapolationType.Extension
            )
        end
        
        # Function that returns all 8 row voltages at time t
        V_rows_func = t -> [v_funcs[row](t) for row in 1:kernel_size]
        
        # Setup ODE parameters
        p = (memristors, V_rows_func, R_series)
        u0 = zeros(Float64, kernel_size * kernel_size)  # 64 initial states
        
        # Solve ODE
        prob = ODEProblem(crossbar_8x8!, u0, (0.0, maximum(times)), p)
        sol = solve(prob, Tsit5(); reltol = 1e-4, abstol = 1e-6, maxiters = Int(1e8))
        
        # Get final states
        X_final = reshape(sol[end], kernel_size, kernel_size)
        
        # Calculate column currents (reservoir outputs)
        V_rows_final = fill(0.1, kernel_size)  # Assume small Check voltage at end
        I_cols, V_cols = calculate_column_currents(
            X_final, 
            V_rows_final, 
            memristors, 
            R_series
        )
        
        # Store features
        feature_start = (i - 1) * kernel_size + 1
        x_vals[feature_start:feature_start+kernel_size-1, j] = Float32.(I_cols) * 1e6  # Convert to µA
    end
    
    # Bias term
    x_vals[end, j] = 1.0
end

using Serialization: serialize
serialize("train_features.jls", x_vals)
test_vals = zeros(Float32, n_total_features, size(test_x, 3))  # (393, 10000)

# Training data processing
@showprogress 1 "Processing test images" for j in 1:size(test_x, 3)
    image_data = test_x[:, :, j]'
    patches = extract_patches(image_data, (kernel_size, kernel_size), stride)  # (49, 8, 8)
    
    for i in 1:size(patches, 1)  # Iterate over all patches
        # Get the i-th patch as 8×8 matrix
        patch = patches[i, :, :]  # Extract 8×8 from 3D array
        
        # Create voltage functions for each ROW
        v_funcs = Vector{AkimaInterpolation}(undef, kernel_size)
        times = nothing
        
        for row in 1:kernel_size
            # Get pixel values from this row
            row_pixels = patch[row, :]  # 8 pixels in this row
            
            # Generate carrier wave: 8 pulses, amplitudes = pixel values
            times, v_signal = generate_carrier_wave(
                1e-5, 
                row_pixels .* 1.5;
                duty_cycle = 0.8, 
                fs = 1e7
            )
            
            # Create interpolation for this row
            v_funcs[row] = AkimaInterpolation(
                v_signal, 
                times; 
                extrapolation = ExtrapolationType.Extension
            )
        end
        
        # Function that returns all 8 row voltages at time t
        V_rows_func = t -> [v_funcs[row](t) for row in 1:kernel_size]
        
        # Setup ODE parameters
        p = (memristors, V_rows_func, R_series)
        u0 = zeros(Float64, kernel_size * kernel_size)  # 64 initial states
        
        # Solve ODE
        prob = ODEProblem(crossbar_8x8!, u0, (0.0, maximum(times)), p)
        sol = solve(prob, Tsit5(); reltol = 1e-4, abstol = 1e-6, maxiters = Int(1e8))
        
        # Get final states
        X_final = reshape(sol[end], kernel_size, kernel_size)
        
        # Calculate column currents (reservoir outputs)
        V_rows_final = fill(0.1, kernel_size)  # Assume small Check voltage at end
        I_cols, V_cols = calculate_column_currents(
            X_final, 
            V_rows_final, 
            memristors, 
            R_series
        )
        
        # Store features
        feature_start = (i - 1) * kernel_size + 1
        test_vals[feature_start:feature_start+kernel_size-1, j] = Float32.(I_cols) * 1e6  # Convert to µA
    end
    
    # Bias term
    test_vals[end, j] = 1.0
end

serialize("test_features.jls", test_vals)


function one_hot_encoding(labels::Vector{Int}, num_classes::Int)
	one_hot = zeros(Float32, num_classes, length(labels))
	for (i, label) in enumerate(labels)
		one_hot[label+1, i] = 1.0
	end
	return one_hot
end

size(x_vals)

y_labels = train_y
y_labels = one_hot_encoding(y_labels, 10)'


y_test_labels = one_hot_encoding(test_y, 10)'

x_vals_t = x_vals'



train_y[2]
y_labels


temp_matrix = rand(Float32, size(x_vals_t, 2), size(y_labels, 2))
x_vals_t * temp_matrix |> size

H = x_vals_t
Y = y_labels
@show size(H)
@show size(Y)




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







W_out = qr_ridge_regression(H, Y, 0.0001)

y_pred = H * W_out
y_pred_labels = map(i -> findmax(y_pred[i, :])[2] - 1, 1:size(y_pred, 1))
accuracy = sum(y_pred_labels .== train_y) / length(train_y)

y_pred_test_labels = map(i -> findmax((test_vals' * W_out)[i, :])[2] - 1, 1:size(test_vals, 2))

test_accuracy = sum(y_pred_test_labels .== test_y) / length(test_y)

println("Training Accuracy: ", accuracy * 100, "%")
println("Test Accuracy: ", test_accuracy * 100, "%")





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







# Function to compute confusion matrix
function compute_confusion_matrix(y_true, y_pred; n_classes=10)
    cm = zeros(Int, n_classes, n_classes)
    for (true_label, pred_label) in zip(y_true, y_pred)
        cm[true_label + 1, pred_label + 1] += 1
    end
    return cm
end

# Compute confusion matrix
cm = compute_confusion_matrix(test_y, y_pred_test_labels)

# Create figure
fig = Figure(size=(800, 700))
ax = Axis(fig[1, 1],
    xlabel="Predicted Label",
    ylabel="True Label",
    title="MNIST Confusion Matrix",
    aspect=DataAspect())



# Plot heatmap
hm = CairoMakie.heatmap!(ax, cm,
    colormap=:Blues,
    colorrange=(0, maximum(cm)))

# Add colorbar
Colorbar(fig[1, 2], hm, label="Count")

# Set ticks to show digits 0-9
ax.xticks = (1:10, string.(0:9))
ax.yticks = (1:10, string.(0:9))

# Add text annotations with counts
for i in 1:10
    for j in 1:10
        count = cm[i, j]
        # Use white text for dark cells, black for light cells
        text_color = count > maximum(cm) / 2 ? :white : :black
        text!(ax, j, i, text=string(count),
            align=(:center, :center),
            color=text_color,
            fontsize=12)
    end
end

# Calculate accuracy
accuracy = sum(diag(cm)) / sum(cm) * 100

# Add accuracy text
text!(ax, 5.5, -0.5,
    text=@sprintf("Overall Accuracy: %.2f%%", accuracy),
    align=(:center, :center),
    fontsize=14,
    font=:bold)

# Save figure
save("mnist_confusion_matrix.png", fig, px_per_unit=2)

# Display figure
fig
# Optional: Print per-class accuracy
println("\nPer-class Accuracy:")
for i in 0:9
    class_total = sum(cm[i+1, :])
    class_correct = cm[i+1, i+1]
    class_acc = class_correct / class_total * 100
    println("Digit $i: $(round(class_acc, digits=2))%")
end

sol_array = Array(sol)  # Each column corresponds to a time point
soll_array = reshape(sol_array, 8, 8, length(t))  # 8×8×time array
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Memristor State X", title = "Memristor States in 8x8 Crossbar Array")
for i in 1:8, j in 1:8
    X_ij = soll_array[i, j, :]
    lines!(ax, t, X_ij, label = "M($i,$j)")
end
ax2 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Voltage (V)", title = "Input Row Voltages")
V_rows_over_time = [V_in(ti) for ti in t]
for i in 1:8
    V_row_i = [V_rows_over_time[k][i] for k in 1:length(t)]
    lines!(ax2, t, V_row_i, label = "V_row($i)")
end
fig 



