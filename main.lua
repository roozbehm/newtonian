-- Usage th main.lua {train|test}

mode = arg[1]
assert (mode=='train' or mode=='test', "Bad arguments. Usage th main.lua {train|test}")

require 'cunn'
-- require 'fbcunn'
require 'cudnn'
require 'xlua'
require 'optim'
require 'math'
require 'gnuplot'
require 'sys'
require 'image'

mattorch = require('fb.mattorch');
pl = require'pl.import_into'()
debugger = require('fb.debugger');

-- fix the random seed for ease of debugging
paths.dofile('setting_options.lua');
cutorch.setDevice(config.GPU);
torch.manualSeed(config.GPU);
----------------------------
paths.dofile('utils.lua');
----------------------------
paths.dofile('data.lua');
----------------------------------
paths.dofile('layers/SmoothPairwiseCosineSimilarity.lua');
-----------------------------
paths.dofile('networks/ModelConstruction_IM_GEFixParallel.lua');
--------------------------------
paths.dofile('train_functions.lua');
------------------------------
log(config)

if mode == 'test' then
	config.nIter = GetVideoCount(testset)
	model:LoadModelFull(config.initModelPath.fullNN)
	log(model.fullNN)
	test()
else
	model:LoadModel(config.initModelPath.imageNN,config.initModelPath.animNN)
	log(model.fullNN)
	train()
end
