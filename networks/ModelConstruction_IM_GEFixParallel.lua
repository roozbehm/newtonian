--Constructing The NN model
log('Constructing Network Model ..... \n');
---------------------------------------

model={};
model.imageNN = require('models.image_row')
model.animationFix = require('models.motion_embedding')
model.animationNN =require('models.motion_row');
model.jointNN_1 = require('models.classifier')
model.jointNN_2 = require('models.pairwisecosine_GEfix')
model.criterion =  nn.ClassNLLCriterion():cuda()

function model:infer(input, k)
  if not model.animFeatures then
    model.animFeatures = GetAnimationFeatures(model.animationNN)
  end
  -- Forward passs
  local tic = os.clock()
  model.fullNN:forward({input, model.animFeatures});
  local toc = os.clock()
  print("Forward time ", tic - toc)
  return model.fullNN.output
end

function model:LearningRateComp(iter)
  local lIter = (iter % config.nResetLR)+1;
  local regimes= config.regimes;
  for _, row in ipairs(regimes) do
    if lIter >= row[1] and lIter <= row[2] then
      return row[3];
    end
  end
end

function model:TrainOneBatch(input,target)
  -- Set into training phase (just active the droputs)
  model.fullNN:training();
  -- Forward passs
  model.fullNN:forward(input);
  
  -- Compute loss and accuracy
  local loss = model.criterion:forward(model.fullNN.output,target)
  local output = model.fullNN.output
  local _, predictedLabel = torch.max(output,2);
  predictedLabel = predictedLabel[{{}, 1}]
  local acc, per_class = GetPerClassAccuracy(predictedLabel, target)
  
  -- Make sure gradients are zero
  model.fullNN:zeroGradParameters();

  -- Backward pass
  local bwCri = model.criterion:backward(model.fullNN.output,target)
  model.fullNN:backward(input,bwCri);

  -- updating the weights
  model.fullNN:updateParameters(model.learningRate);
  return acc,loss;
end

function model:EvaluateOneBatch(input,target)
  -- Set into Evaluation mode (just deactive the dropouts)
  model.fullNN:evaluate();
  local loss = 0;  
  local infer_output = model:infer(input,1);
  local max,predictedLabel = torch.max(infer_output,2);

  predictedLabel = predictedLabel[{{},1}] -- convert matrix to vector

  local _, bestFrame = torch.max(model.jointNN_2:get(2).output, 3)
  local acc, per_class = GetPerClassAccuracy(predictedLabel, target)

  return acc, loss, per_class, predictedLabel
end

function model:SaveModel(fileName)
  local saveModel ={};
  -- reading model parameters to CPU
  saveModel.imageNN       = GetNNParamsToCPU(model.imageNN);
  saveModel.animationNN   = GetNNParamsToCPU(model.animationNN);
  saveModel.animationFix  = GetNNParamsToCPU(model.animationFix);
  saveModel.jointNN_1     = GetNNParamsToCPU(model.jointNN_1);
  saveModel.jointNN_2     = GetNNParamsToCPU(model.jointNN_2);
  -- saving into the file
  torch.save(fileName,saveModel)
end

function model:LoadCaffeImageNN(caffeFilePath)
  local protoFile = caffeFilePath.proto
  local modelFile = caffeFilePath.model
  local meanFile  = caffeFilePath.mean

  require 'loadcaffe'
  local caffeModel = loadcaffe.load(protoFile,modelFile,'cudnn');
  caffeModel:remove(24);
  caffeModel:remove(23);
  caffeModel:remove(22);
  local caffeParams = GetNNParamsToCPU(caffeModel);
  if config.w_crop then
    caffeParams[1] = caffeParams[1]:repeatTensor(1, 5, 1, 1)
  end
  if config.train.mask.enable then
    local firstLayerRandom = torch.FloatTensor(96, 1, 11, 11)
    firstLayerRandom:apply(rand_initialize)
    caffeParams[1] = torch.cat(firstLayerRandom, caffeParams[1], 2)
  end
  LoadNNlParams(model.imageNN, caffeParams);
  
  LoadCaffeMeanStd(meanFile);
end


function model:LoadModel(fileNameImg,fileNameAnim)
  log('Loding Network Model ....')
  for mm = 19,16,-1 do
   model.animationNN:remove(mm);
  end
  model.animationNN:add(nn.Transpose{2,3}):add(nn.Reshape(10,4096,false)  );
  model.animationNN:cuda()

  if fileNameImg == "caffe" then
    model:LoadCaffeImageNN(config.caffeFilePath);
    model.jointNN_1:apply(rand_initialize);
  elseif fileNameImg then
    local saveModel = torch.load(fileNameImg);
    LoadNNlParams(model.imageNN ,saveModel.imageNN);
    LoadNNlParams(model.jointNN_1 ,saveModel.jointNN);
  else
    -- Initialize the model randomly
    model.imageNN:apply(rand_initialize);
  end
  model.jointNN_2:apply(rand_initialize);

  if config.caffeInit then 
      LoadCaffeMeanStd(config.caffeFilePath.mean);
  end

  if fileNameAnim then
    local saveModel = torch.load(fileNameAnim);
    LoadNNlParams(model.animationNN ,saveModel.imageNN);
  else
    -- Initialize the model randomly
    model.animationNN:apply(rand_initialize);
  end
  
  model.animationFix:apply(rand_initialize);
 
  local featuresTable = nn.ParallelTable():add(model.imageNN):add(model.animationFix);
  local classifier    = nn.Sequential():add(nn.SelectTable(1)):add(model.jointNN_1):add(nn.MulConstant(config.lambda,true));
  local matcher       = nn.Sequential():add(model.jointNN_2):add(nn.MulConstant((1-config.lambda),true));
  local concatTable   = nn.ConcatTable():add(classifier):add(matcher)
  model.fullNN = nn.Sequential():add(featuresTable):add(concatTable):add(nn.CAddTable())
  model.fullNN:cuda();

  model:SaveModel( paths.concat(config.logDirectory, 'init.t7'))
end

function model:LoadModelFull(fileName)
  log('Loding Network Model ....')

  for mm = 19,16,-1 do
   model.animationNN:remove(mm);
  end
  model.animationNN:add(nn.Transpose{2,3}):add(nn.Reshape(10,4096,false)  );
  model.animationNN:cuda()

  if fileName  then
    local saveModel = torch.load(fileName);
    LoadNNlParams(model.imageNN ,saveModel.imageNN);
    LoadNNlParams(model.animationNN ,saveModel.animationNN);
    -- debugger.enter()
    LoadNNlParams(model.animationFix ,saveModel.animationFix);
    LoadNNlParams(model.jointNN_1,saveModel.jointNN_1);
    LoadNNlParams(model.jointNN_2,saveModel.jointNN_2);
  else
    -- Initialize the model randomly
    model.imageNN:apply(rand_initialize);
    model.animationNN:apply(rand_initialize);
    model.animationFix:apply(rand_initialize);
    model.jointNN_1:apply(rand_initialize);
    model.jointNN_2:apply(rand_initialize);
  end
  if config.caffeInit then 
      LoadCaffeMeanStd(config.caffeFilePath.mean);
  end
  
  local featuresTable = nn.ParallelTable():add(model.imageNN):add(model.animationFix);
  local classifier    = nn.Sequential():add(nn.SelectTable(1)):add(model.jointNN_1):add(nn.MulConstant(config.lambda,true));
  local matcher       = nn.Sequential():add(model.jointNN_2):add(nn.MulConstant((1-config.lambda),true));
  local concatTable   = nn.ConcatTable():add(classifier):add(matcher)
  model.fullNN = nn.Sequential():add(featuresTable):add(concatTable):add(nn.CAddTable())
  model.fullNN:cuda();

  model:SaveModel( paths.concat(config.logDirectory, 'init.t7'))
end
