using Flux, Flux.Data.MNIST
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
	#println(y)

	x_indices = trunc.(Int, Array(selectdim(x, 1, 1)))
	y_indices = trunc.(Int, Array(selectdim(y, 2, 1)))

	println("*************************************")
	println((x_indices[:, 1]))
	println("//////////////////////////////////////")
	println((y_indices[:, 1]))

	batch = []
	#println(size(img))
	println(batch_size)
	for i in 1:batch_size
		#push!(batch, img[y_indices[:, i], x_indices[:, i], :, i])
		pic = colorview(RGB, reshape(img[:, :, :, i], channels, width, height))
		new_pic = pic[y_indices[:, i], x_indices[:]]
		push!(batch, reshape(Float64.(channelview(new_pic)), width, height, channels))
	end
	batch = reshape(cat(batch..., dims = 4),width, height, channels, batch_size)
	return batch
end


function bilinear_sampler(img, x, y)
	height = size(img)[1]
	width = size(img)[2]
	channels = size(img)[3]
	batch_size = size(img)[4]
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

	new_img = colorview(RGB, reshape(valA[:, :, :, 1], 3, 64, 64))
	imshow(new_img)
	new_img = colorview(RGB, reshape(valB[:, :, :, 1], 3, 64, 64))
	imshow(new_img)
	new_img = colorview(RGB, reshape(valC[:, :, :, 1], 3, 64, 64))
	imshow(new_img)
	new_img = colorview(RGB, reshape(valD[:, :, :, 1], 3, 64, 64))
	imshow(new_img)

	weight1 = []
	weight2 = []
	weight3 = []
	weight4 = []
	for i in 1:channels
		push!(weight1, w1)
		push!(weight2, w2)
		push!(weight3, w3)
		push!(weight4, w4)
	end

	weight1 = permutedims(reshape(cat(weight1..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])
	weight2 = permutedims(reshape(cat(weight2..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])
	weight3 = permutedims(reshape(cat(weight3..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])
	weight4 = permutedims(reshape(cat(weight4..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])

	resultant = weight1 .* valA + weight2 .* valB + weight3 .* valC + weight4 .* valD
	return resultant
end

theta = Matrix{Float64}(I, 2, 3)
#theta[2, 3] = 0.5
# theta[1, 1] = cos(pi/6)
# theta[1, 2] = -sin(pi/6)
# theta[2, 1] = sin(pi/6)
# theta[2, 2] = cos(pi/6)
theta = Array(reshape(transpose(repeat(transpose(theta), 1)), 2, 3, 1))
# println("Verifying : ")
# println(affine_grid_generator(32, 16, theta)[1])
# println("-----------------------------------------------")
# println(affine_grid_generator(32, 16, theta)[2])

#Predicts the 6 parameters of transformation matrix for each image in the batch.
localization_net = Chain(MaxPool((2, 2)), Conv((5, 5), 1 => 20, stride = (1, 1), pad = (0, 0)),
					MaxPool((2, 2)), Conv((5, 5), 20 => 20, stride = (1, 1), pad = (0, 0)),
					x -> reshape(x, :, size(x, 4)),
					Dense(1620, 50), x -> relu.(x),
					Dense(50, 6))
