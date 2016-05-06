require 'cudnn'
require 'cunn'
local alexnet = nn.Sequential()
require 'inn'

local input_channels = 3
if (mode == 'train' and config.train.mask.enable)
   or (mode == 'test' and config.test.mask.enable) then
	input_channels = input_channels + 1
end

alexnet:add(cudnn.SpatialConvolution(input_channels, 96, 11, 11, 4, 4, 0, 0, 1));
alexnet:add(cudnn.ReLU(true))
alexnet:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
alexnet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
alexnet:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2))
alexnet:add(cudnn.ReLU(true))
alexnet:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
alexnet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
alexnet:add(cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, 1))
alexnet:add(cudnn.ReLU(true))
alexnet:add(cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2))
alexnet:add(cudnn.ReLU(true))
alexnet:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2))
alexnet:add(cudnn.ReLU(true))
alexnet:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
alexnet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
alexnet:add(nn.View(-1):setNumInputDims(3))
alexnet:add(nn.Linear(9216, 4096))
alexnet:add(cudnn.ReLU(true))
alexnet:add(nn.Dropout(config.dropoutProb))
alexnet:add(nn.Linear(4096, 4096))
alexnet:add(cudnn.ReLU(true))

return alexnet