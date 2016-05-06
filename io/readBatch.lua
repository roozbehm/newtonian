log ("loading datasets metadata");
dataset_GE = LoadGEDatabase();
trainset   = LoadTrainDatabase(config.excluded_categories);
testset    = LoadTestDatabase(config.excluded_categories);
log ("computing mean_std");
compute_mean_std(trainset, dataset_GE);
log ("dataset done");

GetASiameseBatchCoroutine = coroutine.create(function(nPositiveImages, nDifferentAngleImages, nDifferentCategoryImages, test)
  local dataset = test and testset or trainset; -- TODO(hessam): local or global?
  assert(nPositiveImages > 0);
  local n1, n2, n3 = nPositiveImages, nDifferentAngleImages, nDifferentCategoryImages;
  local batchSize = nPositiveImages + nDifferentAngleImages + nDifferentCategoryImages;
  local target;
  local images;

  local all_input_types = GetEnableInputTypes(dataset.config)
  local nChannels       = GetValuesSum(all_input_types)
  if config.GPU == -1 then -- CPU mode
    target = torch.FloatTensor(batchSize);
    images = torch.FloatTensor(batchSize, nChannels, config.imH, config.imW);
  else
    target = torch.CudaTensor(batchSize);
    images = torch.CudaTensor(batchSize, nChannels, config.imH, config.imW);
  end

  local trainIndex = nil
  local testIndex = nil
  local trainvideos = MakeShuffledTuples(trainset);
  local testvideos = MakeShuffledTuples(testset);

  repeat
    nPositiveImages, nDifferentAngleImages, nDifferentCategoryImages = n1, n2, n3;
    -- Iterate on real videos not game engines to make sure all videos are seen.
    local v
    if test then
      testIndex, v = next(testvideos, testIndex)
      if testIndex == nil then
        testvideos = MakeShuffledTuples(testset);
        testIndex, v = next(testvideos, testIndex)
      end
    else
      trainIndex, v = next(trainvideos, trainIndex)
      if trainIndex == nil then
        trainvideos = MakeShuffledTuples(trainset);
        trainIndex, v = next(trainvideos, trainIndex)
      end
    end

    local batchIndex       = 1;
    local category         = v[1];
    local physicsCategory  = v[2];
    local angle            = v[3];
    local folder           = v[4];

    local gameEngineVideo = LoadGEPerCategory(physicsCategory, angle, dataset_GE);
    
    images[1]  = LoadRandomFrameOfVideo(dataset, category, angle, folder);

    target[1] = 1;
    nPositiveImages = nPositiveImages - 1;
    batchIndex = batchIndex + 1;

    repeat
      local shuffledDataset = MakeShuffledTuples(dataset);
      for _,sample in pairs(shuffledDataset) do
        sampleCategory         = sample[1];
        samplePhysicsCategory  = sample[2];
        sampleAngle            = sample[3];
        sampleFolder           = sample[4];
        if nPositiveImages > 0 and samplePhysicsCategory == physicsCategory
                and sampleAngle == angle then
          -- Add the positive example
          images[batchIndex] = LoadRandomFrameOfVideo(dataset, sampleCategory, sampleAngle, sampleFolder, 'image');
          target[batchIndex] = 1;
          nPositiveImages = nPositiveImages - 1;
          batchIndex = batchIndex + 1;
        elseif nDifferentAngleImages > 0 and samplePhysicsCategory == physicsCategory
                and sampleAngle ~= angle then
          -- Add the negative example with different angle
          images[batchIndex] = LoadRandomFrameOfVideo(dataset, sampleCategory, sampleAngle, sampleFolder, 'image');
          target[batchIndex] = 0;
          nDifferentAngleImages = nDifferentAngleImages - 1;
          batchIndex = batchIndex + 1;
        elseif nDifferentCategoryImages > 0 and samplePhysicsCategory ~= physicsCategory then
          -- Add the negative example with different physics category
          images[batchIndex] = LoadRandomFrameOfVideo(dataset, sampleCategory, sampleAngle, sampleFolder, 'image');
          target[batchIndex] = 0;
          nDifferentCategoryImages = nDifferentCategoryImages - 1;
          batchIndex = batchIndex + 1;
        end
        if batchIndex > batchSize then
          break;
        end
      end

      if nDifferentAngleImages == n2 then -- no different angle exists
        nDifferentCategoryImages = nDifferentCategoryImages + nDifferentAngleImages
        nDifferentAngleImages = 0
      end
    until batchIndex > batchSize

    assert(batchIndex > batchSize, "Not enough data to generate a batch for category="
          .. physicsCategory .. " and angle=" .. tostring(angle) .. ". Requirments = ("
          .. tostring(nPositiveImages) .. "," .. tostring(nDifferentAngleImages) .. ","
          .. tostring(nDifferentCategoryImages) .. ")");
    -- shuffle data
    local shuffle = torch.randperm(batchSize):type('torch.LongTensor')
    images = images:index(1, shuffle)
    target = target:index(1, shuffle)
    local randomForce = torch.random( gameEngineVideo:size(1) )
    local gameEngineVideoRandomForce = gameEngineVideo[{{randomForce}, {}, {}, {}, {}}]:transpose(2, 3)
    if config.GPU ~= -1 then
      gameEngineVideoRandomForce = gameEngineVideoRandomForce:cuda()
    end

    -- yeild the output
    _, _, _, test = coroutine.yield({images, gameEngineVideoRandomForce}, target);
    dataset = test and testset or trainset;
  until false -- repeat until the end of the world
end)

GetAnImageBatchCoroutine = coroutine.create(function(batchSize, useViewPoint, test, deterministic, spline)
  assert(batchSize > 0);
  assert((not spline) or useViewPoint, "Can't get splines with no viewpoint");

  local splinesMat = getmetatable(dataset_GE)['splines']

  local target;
  local images;
  local dataset = test and testset or trainset

  local all_input_types = GetEnableInputTypes(dataset.config)
  local nChannels       = GetValuesSum(all_input_types)

  if config.GPU == -1 then -- CPU mode
    target = spline and torch.FloatTensor(batchSize, splinesMat:size(2)) or torch.FloatTensor(batchSize);
    images = torch.FloatTensor(batchSize, nChannels, config.imH, config.imW);
  else
    target = spline and torch.CudaTensor(batchSize, splinesMat:size(2)) or torch.CudaTensor(batchSize);
    images = torch.CudaTensor(batchSize, nChannels, config.imH, config.imW);
  end

  local batchIndex = 1;
  repeat
    local videos = MakeShuffledTuples(dataset, deterministic);
    for _,v in pairs(videos) do
      local category         = v[1];
      local physicsCategory  = v[2];
      local angle            = v[3];
      local folder           = v[4];
      local categoryId
      if (useViewPoint) then
        categoryId = GetCategoryViewPointId(physicsCategory, angle);
      else
        categoryId = GetCategoryId(physicsCategory);
      end
      images[batchIndex] = LoadRandomFrameOfVideo(dataset, category, angle, folder);
      target[batchIndex] = spline and splinesMat[categoryId] or categoryId

      batchIndex = batchIndex + 1
      if batchIndex > batchSize then
        if config.GPU ~= -1 then
          images:cuda()
          target:cuda()
        end
        _, _, test, deterministic, _ = coroutine.yield(images, target);
        -- re-initialize vars for the next batch:
        dataset = test and testset or trainset
        batchIndex = 1
      end
    end
  until false -- repeat until the end of the world
end)

GetAUniformImageBatchCoroutine = coroutine.create(function(batchSize, useViewPoint, test, spline)
  assert(batchSize > 0);
  assert((not spline) or useViewPoint, "Can't get splines with no viewpoint");

  local splinesMat = getmetatable(dataset_GE)['splines']

  local target;
  local images;
  local dataset = test and testset or trainset

  local all_input_types = GetEnableInputTypes(dataset.config)
  local nChannels       = GetValuesSum(all_input_types)
  if config.GPU == -1 then -- CPU mode
    target = spline and torch.FloatTensor(batchSize, splinesMat:size(2)) or torch.FloatTensor(batchSize);
    images = torch.FloatTensor(batchSize, nChannels, config.imH, config.imW);
  else
    target = spline and torch.CudaTensor(batchSize, splinesMat:size(2)) or torch.CudaTensor(batchSize);
    images = torch.CudaTensor(batchSize, nChannels, config.imH, config.imW);
  end

  repeat
    local batchIndex = 1
    dataset = test and testset or trainset
    repeat
      local randomData = GetUniformRandomData(dataset)
      local category = randomData[1]
      local physicsCategory = randomData[2]
      local angle = randomData[3]
      local folder = randomData[4]
      if (useViewPoint) then
        categoryId = GetCategoryViewPointId(physicsCategory, angle);
      else
        categoryId = GetCategoryId(physicsCategory);
      end
      images[batchIndex] = LoadRandomFrameOfVideo(dataset, category, angle, folder)
      target[batchIndex] = spline and splinesMat[categoryId] or categoryId

      batchIndex = batchIndex + 1
    until batchIndex > batchSize

    if config.GPU ~= -1 then
      images:cuda()
      target:cuda()
    end
    _, _, test, _ = coroutine.yield(images, target);
  until false -- repeat until the end of the world
end)

GetAUniformAnimationBatchCoroutine = coroutine.create(function(batchSize, useViewPoint, spline)
  assert(batchSize > 0);
  assert((not spline) or useViewPoint, "Can't get splines with no viewpoint");

  local splinesMat = getmetatable(dataset_GE)['splines']

  local nChannels = GetValuesSum(GetEnableInputTypes(config.GE))
  local target;
  local videos;
  if config.GPU == -1 then -- CPU mode
    target = spline and torch.FloatTensor(batchSize, splinesMat:size(2)) or torch.FloatTensor(batchSize);
    videos = torch.FloatTensor(batchSize, nChannels, config.GE.frame_per_video, config.GE.imH, config.GE.imW);
  else
    target = spline and torch.CudaTensor(batchSize, splinesMat:size(2)) or torch.CudaTensor(batchSize);
    videos = torch.CudaTensor(batchSize, nChannels, config.GE.frame_per_video, config.GE.imH, config.GE.imW);
  end

  repeat
    local batchIndex = 1;
    repeat
      local randomCategoryIndex = torch.random( #config.classes )
      local physicsCategory = config.classes[ randomCategoryIndex ]
      local angle = torch.random( config.class_angles[randomCategoryIndex] )
      if (useViewPoint) then
        categoryId = GetCategoryViewPointId(physicsCategory, angle)
      else
        categoryId = GetCategoryId(physicsCategory)
      end
      local gameEngineVideo = LoadGEPerCategory(physicsCategory, angle, dataset_GE)
      local gameEngineVideoRandomForce = gameEngineVideo[ torch.random(gameEngineVideo:size(1)) ]
      videos[batchIndex]  = gameEngineVideoRandomForce:transpose(1,2);
      target[batchIndex]  = spline and splinesMat[categoryId] or categoryId

      batchIndex = batchIndex + 1
    until batchIndex > batchSize

    if config.GPU ~= -1 then
      videos:cuda()
      target:cuda()
    end
    coroutine.yield(videos, target);
  until false -- repeat until the end of the world
end)

GetAVideoBatchCoroutine = coroutine.create(function(useViewPoint, test, spline)
  assert((not spline) or useViewPoint, "Can't get splines with no viewpoint");

  local dataset = test and testset or trainset

  local splinesMat = getmetatable(dataset_GE)['splines']

  repeat
    local videos = MakeShuffledTuples(dataset);
    for _,v in pairs(videos) do
      local category         = v[1];
      local physicsCategory  = v[2];
      local angle            = v[3];
      local folder           = v[4];
      local categoryId
      if (useViewPoint) then
        categoryId = GetCategoryViewPointId(physicsCategory, angle);
      else
        categoryId = GetCategoryId(physicsCategory);
      end

      local video   = LoadTrainImagesPerVideo(dataset, category, angle, folder);
      local target  = spline and splinesMat[categoryId] or categoryId
      if config.GPU ~= -1 then
        video = video:cuda()
      end
      _, test, _ = coroutine.yield(video, target);
      -- re-initialize vars for the next batch:
      dataset = test and testset or trainset
    end
  until false -- repeat until the end of the world
end)
