  function RemoveDotDirs(aTable)
  if aTable == nil or type(aTable) ~= 'table' then
    return aTable
  end
  --remove the two directories "." , ".."
  local i = 1
  while i <= #aTable do
    while aTable[i] ~= nil and aTable[i]:sub(1,1) == '.' do
      aTable[i] = aTable[#aTable]
      aTable[#aTable] = nil
    end
    i = i + 1
  end
end

function getTableSize(aTable)
  local numItems = 0
  for k,v in pairs(aTable) do
      numItems = numItems + 1
  end
  return numItems
end

function GetRandomValue(aTable)
  local values = {}
  for key, value in pairs(aTable) do
    values[ #values+1 ] = value
  end
  return values[ torch.random(#values) ]
end

function GetValuesSum(aTable)
  local total = 0
  for key, value in pairs(aTable) do
    total = total + value
  end
  return total
end

function loadImageOrig(path)
  -----------------------------------------------------------------
  -- Reads an image
  -- inputs:
  --        "path": path to the image
  -- output:
  --        "im": the image
  -----------------------------------------------------------------
   local im = image.load(path)
      if im:dim() == 2 then -- 1-channel image loaded as 2D tensor
      im = im:view(1,im:size(1), im:size(2)):repeatTensor(3,1,1)
   elseif im:dim() == 3 and im:size(1) == 1 then -- 1-channel image
      im = im:repeatTensor(3,1,1)
   elseif im:dim() == 3 and im:size(1) == 3 then -- 3-channel image
   elseif im:dim() == 3 and im:size(1) == 4 then -- image with alpha
      im = im[{{1,3},{},{}}]
   else
      error("image structure not compatible")
   end
   return im
end

function loadImage(path, imH, imW)
  -----------------------------------------------------------------
  -- Reads an image and rescales it
  -- inputs:
  --        "path": path to the image
  --        "imH" and "imW": the image is rescaled to imH x imW
  -- output:
  --        "im": the rescaled image
  -----------------------------------------------------------------
   local im = loadImageOrig(path)
   im = image.scale(im, imW, imH)
   return im
end

function normalizeImage(im, mean, std)
  -----------------------------------------------------------------
  -- Normalizes image "im" by subtracting the "mean" and dividing by "std"
  -----------------------------------------------------------------
  for channel=1,3 do
    im[{channel,{},{}}]:add(-mean[channel]);
    im[{channel,{},{}}]:div(std[channel]);
  end
  return im;
end

function LoadRandomSamples(nSamples, allfiles, imH, imW);
  -----------------------------------------------------------------
  -- Loads "nSamples" images from the "allfiles" and rescaled them to imH x imW
  -- inputs:
  --       nSamples: # of images that is sampled
  --       allfiles: an array of paths of the images in the dataset
  --       imH, imW: size of the rescaled image
  -- outputs:
  --       images: 4D Tensor that includes "nSamples" number of imHximW images
  -----------------------------------------------------------------
  local images = torch.Tensor(nSamples, 3, imH, imW);
  local randnums = torch.randperm(#allfiles);
  local idx = randnums[{{1,nSamples}}];
  for i = 1,nSamples do
    local fname = allfiles[idx[i]];
    local im = loadImage(fname, imH, imW);
    images[{{i},{},{},{}}] = im;
  end
  return images;
end

function ComputeMeanStd(nSample, allfiles, imH, imW)
  -----------------------------------------------------------------
  -- Computes the mean and std of randomly sampled images
  -- inputs:
  --       nSample: # of images that is sampled
  --       allfiles: an array of paths of the images in the dataset
  --       imH, imW: size of the rescaled image
  -- outputs:
  --       mean: a 3-element array (the mean for each channel)
  --       std:  a 3-element array (the std for each channel)
  -----------------------------------------------------------------

  local images    = LoadRandomSamples(nSample, allfiles, imH, imW);
  local mean = {};
  local std  = {};

  mean[1]   = torch.mean(images[{{},1,{},{}}]);
  mean[2]   = torch.mean(images[{{},2,{},{}}]);
  mean[3]   = torch.mean(images[{{},3,{},{}}]);

  std[1]    = torch.std(images[{{},1,{},{}}]);
  std[2]    = torch.std(images[{{},2,{},{}}]);
  std[3]    = torch.std(images[{{},3,{},{}}]);

  return mean, std;
end

function MakeListTrainFrames(dataset, trainDir, image_type)
  allfiles = {};
  for category, subdataset in pairs(dataset) do
    if category ~= 'config' then
      for angles, subsubdataset in pairs(subdataset) do
        for dirs, files in pairs(subsubdataset) do
          for _, f in pairs(files) do
            fname = string.sub(f, 1, -11) .. "." .. image_type;
            table.insert(allfiles, paths.concat(trainDir, category, dirs, fname));
          end
        end
      end
    end
  end
  return allfiles;
end

function MakeListGEFrames(dataset, data_type)
  local geDir   = config.GE.dir;
  allfiles = {};
  for categories, subdataset in pairs(dataset) do
    for angles, subsubdataset in pairs(subdataset) do
      for dirs, files in pairs(subsubdataset) do
        for _, f in pairs(files) do
          table.insert(allfiles, paths.concat(geDir, categories, categories .. "_" .. angles .. "_" .. data_type, dirs, f));
        end
      end
    end
  end
  return allfiles;
end

function shuffleList(list, deterministic)
  local rand
  if deterministic then -- shuffle! but deterministicly.
    math.randomseed(2)
    rand = math.random
  else
    rand = torch.random
  end

  for i = #list, 2, -1 do
      local j = rand(i)
      list[i], list[j] = list[j], list[i]
  end
end

function GetPhysicsCategory(category)
  return category:match("[^-]+")
end

function MakeShuffledTuples(dataset, deterministic)
  -- tuple: category, physics category, angle, folder
  local trainDir   = config.trainDir;
  tuples = {};
  for category, subdataset in pairs(dataset) do
    if category ~= 'config' then
      local physicsCategory = GetPhysicsCategory(category)
      for angles, subsubdataset in pairs(subdataset) do
        for dirs, _ in pairs(subsubdataset) do
          table.insert(tuples, {category, physicsCategory, angles, dirs});
        end
      end
    end
  end
  shuffleList(tuples, deterministic);
  return tuples;
end

function isExcluded(excluded_categories, category)
  for _, ecat in pairs(excluded_categories) do
    if category:find(ecat) then
      return true
    end
  end
  return false
end

function removeExcludedCategories(categories, excluded_categories)
  local result = {};
  for k,v in pairs(categories) do
    if not isExcluded(excluded_categories, v) then
      table.insert(result, v);
    end
  end
  assert(#result + #excluded_categories <= #categories, "At least one category" ..
                  "should be removed per excluded_categories.")
  assert(#result > 0, "Cannot exclude all categories.")
  return result;
end

function getAllCategoriesandAngles(dataset)
  physics_category_list = {};
  category_list = {};
  angle_list = {};
  for k,v in pairs(dataset) do
    table.insert(physics_category_list, GetPhysicsCategory(k))
    table.insert(category_list, k);
    table.insert(angle_list, getTableSize(dataset[k]));
  end
  return physics_category_list, category_list, angle_list;
end

function GetNNParamsToCPU(nnModel)
  -- Convert model into FloatTensor and save.
  local params, gradParams = nnModel:parameters()
  if params ~= nill then
    paramsCPU = pl.tablex.map(function(param) return param:float() end, params)
  else
    paramsCPU = {};
  end
  return paramsCPU
end

function LoadNNlParams(current_model,saved_params)
  local params, gradparams = current_model:parameters()
  if params ~= nill then
    assert(#params == #saved_params,
      string.format('#layer != #saved_layers (%d vs %d)!',
        #params, #saved_params));
    for i = 1,#params do
      assert(params[i]:nDimension() == saved_params[i]:nDimension(),
        string.format("Layer %d: dimension mismatch (%d vs %d).",
          i, params[i]:nDimension(), saved_params[i]:nDimension()))
      for j = 1, params[i]:nDimension() do
        assert(params[i]:size(j) == saved_params[i]:size(j),
          string.format("Layer %d, Dim %d: size does not match (%d vs %d).",
            i, j, params[i]:size(j), saved_params[i]:size(j)))
      end
      params[i]:copy(saved_params[i]);
    end
  end
end

function rand_initialize(layer)
  local tn = torch.type(layer)
  if tn == "cudnn.SpatialConvolution" then
    local c  = math.sqrt(10.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "cudnn.VolumetricConvolution" then
    local c  = math.sqrt(10.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.Linear" then
    local c =  math.sqrt(10.0 / layer.weight:size(2));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  end
end

function GetCategoryViewPointId(physicsCategory, viewpoint)
  local offset = 0;
  for i, class in ipairs(config.classes) do
    if class == physicsCategory then
      return offset + viewpoint
    end
    offset = offset + config.class_angles[i];
  end
  error("failed to find the physicsCategory:" .. physicsCategory);
  return -1; -- invalid physics category
end

function DecryptCategoryViewPointId(categoryId)
  assert(categoryId > 0, "Invalid categoryId " .. tostring(categoryId))

  local offset = 0;
  for i, class in ipairs(config.classes) do
    if offset + config.class_angles[i] >= categoryId then
      return class, categoryId - offset
    end
    offset = offset + config.class_angles[i];
  end
  error("Invalid categoryId " .. tostring(categoryId));
end

function GetCategoryId(physicsCategory)
  for i, class in pairs(config.classes) do
    if class == physicsCategory then
      return i
    end
  end
  error("failed to find the physicsCategory:" .. physicsCategory);
  return -1; -- invalid physics category
end

function CategoryViewPointId2CategoryId(categoryId)
  assert(categoryId > 0, "Invalid categoryId " .. tostring(categoryId))

  local offset = 0;
  for i, class in ipairs(config.classes) do
    if offset + config.class_angles[i] >= categoryId then
      return i
    end
    offset = offset + config.class_angles[i];
  end
  error("Invalid categoryId " .. tostring(categoryId));
end

function GetUniformRandomElement(data)
  local result = {}
  while type(data) == 'table' do
    local keys = {}
    for key, value in pairs(data) do
      if key ~= 'config' and (type(value) ~= 'table' or next(value) ~= nil) then
        keys[ #keys+1 ] = key
      end
    end
    local randomKey = keys[torch.random(#keys)]
    data = data[randomKey]
    result[#result+1] = randomKey
  end
  result[#result+1] = data
  return result
end

function GetUniformRandomCategory(dataset, physicsCategory, angle)
  local keys = {}
  for key, value in pairs(dataset) do
    if string.sub(key,1,string.len(physicsCategory)) == physicsCategory then
      if value[angle] and next(value[angle]) then
        keys[ #keys+1 ] = key
      end
    end
  end
  if next(keys) then
    return keys[torch.random(#keys)]
  else
    return nil
  end
end

function GetUniformRandomData(dataset)
  local randomData = GetUniformRandomElement(dataset)
  local category = randomData[1]
  local physicsCategory = GetPhysicsCategory(category)
  local angle = randomData[2]
  local folder = randomData[3]
  return {category, physicsCategory, angle, folder}
end

function log(...)
  -- Log to file:
  io.output(config.logFile)
  print(...)
  -- Log to stdout:
  io.output(io.stdout)
  print(...)
end

function GetEnableInputTypes(input_config)
  local result = {}
  for input_type, conf in pairs(input_config) do
    if type(conf) == 'table' and conf.enable then
      if config.w_crop and conf.croppable then
        result[ input_type ] = conf.nChannels * 5
      else
        result[ input_type ] = conf.nChannels
      end
    end
  end
  return result
end

function GetPerClassAccuracy(predictions, labels)
  local per_class = torch.Tensor(config.nCategories, 2):fill(0)
  local nAccurate = 0
  labels = labels:clone()
  predictions = predictions:clone()
  for i=1,labels:size(1) do
    if labels[i] == predictions[i] then
      nAccurate = nAccurate + 1
      per_class[ labels[i] ][1] = per_class[ labels[i] ][1] + 1
    end
    per_class[ labels[i] ][2] = per_class[ labels[i] ][2] + 1
  end
  local acc = nAccurate / labels:size(1)
  return acc, per_class
end

function GetAnimationFeatures(model, convLayer)  
  local n = GetValuesSum(config.class_angles) -- Total number of classes
  local feats
  local labels = {}
  for i=1,n do
    local featsDir = paths.concat(config.GE.featsDir, i)
    local featFiles = paths.dir(featsDir)
    RemoveDotDirs( featFiles )
    if not featFiles or #featFiles==0 then
      log("Animation vectors for category " .. tostring(i) .. " not found.")
      os.execute('mkdir -p ' .. featsDir)

      local category, angle = DecryptCategoryViewPointId(i)
      local gameEngineVideos = LoadGEPerCategory(category, angle, dataset_GE):transpose(2, 3):cuda()
      log("Feed-forward animation to get features.")
      for j=1,gameEngineVideos:size(1) do
        local cur = model:forward( gameEngineVideos[ {{j}, {}, {}, {}, {}} ] )
        if feats then
          feats = torch.cat(feats, cur, 3)
        else
          feats = cur
        end

        for k=1,cur:size(1) do
          labels[ #labels+1 ] = i
        end
        -- Cache for future use:
        torch.save( paths.concat(featsDir, tostring(j) .. '.t7'), cur)
      end
    else
      for j, v in pairs(featFiles) do
        local cur = torch.load( paths.concat(featsDir, v) )
        if feats then
          feats = torch.cat(feats, cur, 3)
        else
          feats = cur
        end
        for k=1,cur:size(1) do
          labels[ #labels+1 ] = i
        end
      end
    end
  end
  feats = feats:transpose(2, 3):transpose(1, 2)
  if convLayer then
    feats = convLayer:forward(feats):reshape(config.nClasses, 10, 4096)
    torch.save(paths.concat( config.DataRootPath, 'all.t7'), feats)
  end
  return feats, labels
end

function GetPairwiseCosine(M1, M2)
  assert(M1:size(2) == M2:size(2), "ERROR: dimensions mismatch!")
  local smooth = 1e-5

  local M1rownorms = torch.cmul(M1, M1):sum(2):sqrt():view(M1:size(1))
  local M2rownorms = torch.cmul(M2, M2):sum(2):sqrt():view(M2:size(1))
  local pairwiseNorms = torch.ger(M1rownorms, M2rownorms)
  local dot = M1 * M2:t()
  return torch.cdiv(dot, pairwiseNorms + smooth)
end

function GetVideoCount(dataset)
  local total = 0
  for _1, cat in pairs(dataset) do
    if _1 ~= 'config' then
      for _2, view in pairs(cat) do
        for _3, fold in pairs(view) do
          total = total + 1
        end
      end
    end
  end
  return total
end

function Choose(tensor, indices)
  assert(tensor:size(1) == indices:size(1), "Dimension mismatch")
  local result = torch.Tensor( indices:size() )
  for i = 1, indices:size(1) do
    result[i] = tensor[i][ indices[i] ]
  end
  return result:cuda()
end

function ContainsValue(dict, value)
  for k,v in pairs(dict) do
    if v == value then
      return true
    end
  end
  return false
end

function GetGaussianTarget(target)
  local result = torch.CudaTensor(target:size(1), config.nClasses):fill(0)
  local frames = target - (torch.floor((target-1) / 10) * 10)
  for i=1,target:size(1) do
    local sigma = 1
    for j = target[i]-frames[i]+1,target[i]-frames[i]+10 do
      result[i][j] = torch.exp( -(target[i] - j)^2 / sigma)
    end

    result[i] = result[i] / result[i]:sum()
  end
  return result
end
