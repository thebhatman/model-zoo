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
	x_s, y_s = affine_grid_generator(out_height, out_width, theta)

end

function affine_grid_generator(height, width, theta)
	batch_size = size(theta)[3]
	x = LinRange(-1, 1, width)
	y = LinRange(-1, 1, height)
	x_t_flat = reshape(repeat(x, height), 1, height*width)
	y_t_flat = reshape(repeat(transpose(y), width), 1, height*width)
	all_ones = ones(eltype(x_t_flat), 1, size(x_t_flat)[2])

	sampling_grid = vcat(x_t_flat, y_t_flat, all_ones)
	sampling_grid = Array(reshape(transpose(repeat(transpose(sampling_grid), batch_size)), 3, size(x_t_flat)[2], batch_size))

	batch_grids = batched_gemm('N', 'N', theta, sampling_grid)
	y_s = permutedims(reshape(batch_grids[2, :, :], width, height, batch_size), [2,1,3])
	x_s = permutedims(reshape(batch_grids[1, :, :], width, height, batch_size), [2,1,3])
	return x_s, y_s
end

function get_pixel_values(img, x, y)
	batch_size = size(x, 3)
	width = size(img, 1)
	height = size(img, 2)
	channels = size(img, 3)
	x_indices = trunc.(Int, Array(selectdim(x, 1, 1)))
	y_indices = trunc.(Int, Array(selectdim(y, 2, 1)))
	batch = []
	println(size(img))
	for i in 1:batch_size
		push!(batch, img[x_indices[:, i], y_indices[:, i],:, i])
	end
	batch = reshape(hcat(batch...),width, height, channels, batch_size)
	return batch
end


function bilinear_sampler(img, x, y)
	height = size(img)[1]
	width = size(img)[2]
	max_y = height
	max_x = width
	x = 0.5*(x .+ 1.0)*(max_x)
	y = 0.5*(y .+ 1.0)*(max_y)

	x0 = trunc.(Int, x)
	x1 = x0 .+ 1.0
	y0 = trunc.(Int, y)
	y1 = y0 .+ 1.0
	x0 = clamp.(x0, 1, max_x)
	x1 = clamp.(x1, 1, max_x)
	y0 = clamp.(y0, 1, max_y)
	y1 = clamp.(y1, 1, max_y)

	w1 = (x1 - x).*(y1 - y)
	w2 = (x1 - x).*(y - y0)
	w3 = (x - x0).*(y1 - y)
	w4 = (x - x0).*(y - y0)

	valA = get_pixel_values(img, x0, y0)
	valB = get_pixel_values(img, x0, y1)
	valC = get_pixel_values(img, x1, y0)
	valD = get_pixel_values(img, x1, y1)


end

theta = Matrix{Float64}(I, 2, 3)
theta = Array(reshape(transpose(repeat(transpose(theta), 10)), 2, 3, 10))
println("Verifying : ")
println(affine_grid_generator(32, 16, theta)[1])
println("-----------------------------------------------")
println(affine_grid_generator(32, 16, theta)[2])
