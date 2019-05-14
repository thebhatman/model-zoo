using Flux
using BatchedRoutines
using LinearAlgebra

function transformer(input_feature, theta, out_dims)
	batch_size = size(input_feature)[4]
	height = size(input_feature)[1]
	width = size(input_feature)[2]
	channels = size(input_feature)[3]

	out_height = out_dims[1]
	out_width = out_dims[2]
	batch_grids = affine_grid_generator(out_height, out_width, theta)
	x_s = batch_grids[:, :, 1, :]
	y_s = batch_grids[:, :, 2, :]

	
end

function affine_grid_generator(height, width, theta)
	batch_size = size(theta)[3]
	x = LinRange(-1, 1, width)
	y = LinRange(-1, 1, height)
	x_t_flat = reshape(repeat(x, height), 1, height*width)
	y_t_flat = reshape(repeat(transpose(y), width), 1, height*width)
	# x_t = reshape(repeat(transpose(x), size(y)[1]), width, height)
	# y_t = reshape(repeat(y, size(x)[1]), height, width)

	# println("y_t ************** ")
	# println((y_t))
	# print("Size of y_t = ", size(y_t))

	# x_t_flat = reshape(x_t, 1, height*width)
	# y_t_flat = reshape(y_t, 1, height*width)
	# println("X_t FLAT ************** ")
	# @show (y_t_flat)
	# print("Size of y_t_flat = ", size(y_t_flat))

	all_ones = ones(eltype(x_t_flat), 1, size(x_t_flat)[2])

	sampling_grid = vcat(x_t_flat, y_t_flat, all_ones)
	sampling_grid = Array(reshape(transpose(repeat(transpose(sampling_grid), batch_size)), 3, size(x_t_flat)[2], batch_size))

	batch_grids = batched_gemm('N', 'N', theta, sampling_grid)
	batch_grids = reshape(batch_grids,2, height, width, batch_size)
	batch_grids = permutedims(batch_grids,[3, 2, 1, 4])
	return batch_grids
end

function bilinear_sampler(img, x, y)
	height = size(img)[1]
	width = size(img)[2]
	max_y = height - 1
	max_x = width - 1
	x = 0.5*(x .+ 1.0)*(max_x)
	y = 0.5*(y .+ 1.0)*(max_y)

	x0 = trunc.(Int, x)
	x1 = x0 .+ 1.0
	y0 = trunc.(Int, y)
	y1 = y0 .+ 1.0
	x0 = clamp.(x0, 0, max_x)
	x1 = clamp.(x1, 0, max_x)
	y0 = clamp.(y0, 0, max_y)
	y1 = clamp.(y1, 0, max_y)

	w1 = (x1 - x).*(y1 - y)
	w2 = (y1 - y).*(x - x0)
	w3 = (y - y0).*(x - x0)
	w4 = (y - y0).*(x1 - x)



end

theta = Matrix{Float64}(I, 2, 3)
theta = Array(reshape(transpose(repeat(transpose(theta), 10)), 2, 3, 10))
println(affine_grid_generator(32, 32, theta)[:,:,1, 1])






