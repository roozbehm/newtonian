local mlp=nn.Sequential()

mlp:add(nn.Reshape(config.nCategories,1,10,4096))

arg[6] = "10*4096FC_1_bn"

local m, var, bnorm = arg[6]:match("([^_]+)_([^_]+)_([^_]+)")

if m=="2*convolve" then
	mlp:add(cudnn.SpatialConvolution(1,10,1001,1,1,1,500,0))
	mlp:add(nn.ReLU(true))
	mlp:add(cudnn.SpatialConvolution(10,20,1,7,1,1,0,3))
elseif m=="10*4096FC" then
	mlp:add(nn.Reshape(config.nCategories*10,4096,false))
	mlp:add(nn.Linear(4096,4096))
	if var == "2" then
		mlp:add(nn.ReLU(true))
		mlp:add(nn.Linear(4096,4096))
	end
	mlp:add(nn.Reshape(config.nCategories,1,10,4096,false))
elseif m=="4096*10FC" then
	mlp:add(nn.Transpose{3,4})
	mlp:add(nn.Reshape(config.nClasses*4096, 10, false))
	mlp:add(nn.Linear(10,10))
	if var == "2" then
		mlp:add(nn.ReLU(true))
		mlp:add(nn.Linear(10,10))
	end
	mlp:add(nn.Reshape(config.nClasses, 1, 4096, 10, false))
	mlp:add(nn.Transpose{3,4})
else
	mlp:add(cudnn.SpatialConvolution(1,20,1,7,1,1,0,3))
end

mlp:add(nn.Max(2))
mlp:add(nn.ReLU(true))
mlp:add(nn.Reshape(config.nCategories*10,4096,false))

if bnorm == "bn" then
	mlp:add(nn.BatchNormalization(4096, 1e-3))
end

return mlp;