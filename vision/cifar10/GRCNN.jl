using Flux, Metalhead, NNlib
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition

Conv_Forward(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv(kernel_size, in_channels => out_channels,relu, pad = pad, stride = stride), BatchNorm(out_channels))

Conv_Rec(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv(kernel_size, out_channels => out_channels, pad = pad, stride = stride), BatchNorm(out_channels))

Conv_Gate_Rec(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv((1,1), out_channels => out_channels, pad = (0,0), stride = (1,1)), BatchNorm(out_channels))

Conv_Gate_Forward(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv(kernel_size, in_channels => out_channels, pad = pad, stride = stride), BatchNorm(out_channels))

Gate(x) = sigmoid(Conv_Gate_Forward(x) + Conv_Gate_Rec(x))

Gate_Mul(x) = Conv_Rec(x)*Gate(x)

function GRCL(x, in_channels, out_channels, kernel_size, pad, stride, n_iter)
	c_f = Conv_Forward(x)
	x = relu.(c_f)
	for i = 1:n_iter
		x = relu.(c_f + BatchNorm(out_channels)(Gate_Mul(x)))
	end
end	

function GRCNN()
	return Chain(Conv((3,3), 1=>64, relu, pad = (1,1), stride = (1,1)),
		BatchNorm(64),
		x -> maxpool(x, (2, 2), pad = (0,0), stride = (2,2)),
		x -> GRCL(x, 64, 64, 3, 1, 1),
		x -> maxpool(x, (2, 2), pad = (0, 0), stride = (2, 2)),
		x -> GRCL(x, 64, 128, 3, 1, 1),
		x -> maxpool(x, (2, 2), pad = (0, 1), stride = (2, 1)),
		x -> GRCL(x, 128, 256, 3, 1, 1),
		x -> maxpool(x, (2, 2), pad = (0, 1), strie = (2, 1)),
		Conv((2,2), 256 => 512, relu, pad = (0,0), stride = (1,1)),
		BatchNorm(512))
end