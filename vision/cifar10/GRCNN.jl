using Flux, Metalhead, NNlib, Flux.Data.MNIST
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition

Conv_Forward(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv(kernel_size, in_channels => out_channels,relu, pad = pad, stride = stride), BatchNorm(out_channels))

Conv_Rec(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv(kernel_size, out_channels => out_channels, pad = pad, stride = stride), BatchNorm(out_channels))

Conv_Gate_Rec(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv((1,1), out_channels => out_channels, pad = (0,0), stride = (1,1)), BatchNorm(out_channels))

Conv_Gate_Forward(in_channels, out_channels, kernel_size, pad, stride) = Chain(Conv(kernel_size, in_channels => out_channels, pad = pad, stride = stride), BatchNorm(out_channels))

#Gate(x, in_channels, out_channels, kernel_size, pad, stride) = sigmoid.(Conv_Gate_Forward(in_channels, out_channels, (kernel_size, kernel_size), (pad, pad), (stride, stride))(x) + Conv_Gate_Rec(in_channels, out_channels, (kernel_size, kernel_size), (pad, pad), (stride, stride))(x))

#Gate_Mul(x, in_channels, out_channels, kernel_size, pad, stride) = (Conv_Rec(in_channels, out_channels, (kernel_size, kernel_size), (pad,pad), (stride,stride))(x)) .* Gate(x, in_channels, out_channels, kernel_size, pad, stride)

forward1 = LSTM(512, 256)
backward1 = LSTM(512, 256)
output1 = Dense(512, 256)

BLSTM1(x) = output1.(vcat.(forward1.(x), Flux.flip(backward1, x)))

forward2 = LSTM(256, 256)
backward2 = LSTM(256, 256)
output2 = Dense(512, 10)

BLSTM2(x) = output2.(vcat.(forward2.(x), flip(backward2, x)))

function GRCL(x, in_channels, out_channels, kernel_size, pad, stride, n_iter)
	conv_forw = Conv_Forward(in_channels, out_channels, (kernel_size, kernel_size), (pad,pad), (stride,stride))(x)
	conv_gate_forw = Conv_Gate_Forward(in_channels, out_channels, (kernel_size, kernel_size), (pad,pad), (stride,stride))(x)
	x = relu.(conv_forw)
	for i = 1:n_iter
		conv_gate_rec = Conv_Gate_Rec(in_channels, out_channels, (kernel_size, kernel_size), (pad,pad), (stride,stride))(x)
		gate = sigmoid.(conv_gate_forw + conv_gate_rec)
		conv_rec = Conv_Rec(in_channels, out_channels, (kernel_size, kernel_size), (pad,pad), (stride,stride))(x)
		x = relu.(conv_forw + BatchNorm(out_channels)(conv_rec .* gate))
	end
	return x
end	

function GRCNN()
	return Chain(Conv((3,3), 1=>64, relu, pad = (1,1), stride = (1,1)),
		BatchNorm(64),
		x -> maxpool(x, (2, 2), pad = (0,0), stride = (2,2)),
		x -> GRCL(x, 64, 64, 3, 1, 1, 3),
		x -> maxpool(x, (2, 2), pad = (0, 0), stride = (2, 2)),
		x -> GRCL(x, 64, 128, 3, 1, 1, 3),
		x -> maxpool(x, (2, 2), pad = (0, 1), stride = (2, 1)),
		x -> GRCL(x, 128, 256, 3, 1, 1, 3),
		x -> maxpool(x, (2, 2), pad = (0, 1), stride = (2, 1)),
		Conv((2,2), 256 => 512, relu, pad = (0,0), stride = (1,1)),
		BatchNorm(512),
		x -> reshape(x, :, size(x, 4)),
		x -> BLSTM1(x),
		x -> BLSTM2(x)
		)
end

x = rand(32, 32, 1, 100)
GRCNN()(x)
