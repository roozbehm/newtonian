function LoadDatabase(dataConfig, excluded_categories)
  -----------------------------------------------------------------
  -- Reads the list of images in the videos from annotDir
  -- inputs:
  --       dataConfig: The data configuration to load from. Look at config.train
  --       and config.test.
  --       exclude_category: exclude this category in training
  -- outputs:
  --       dataset: a table with list of files for each category
  --       dataset[category][angle][video_directory]
  --       e.g., dataset['sliding-ski'][1]["181_1"] contains the files
  --       for video "181_1", which is annotated as the first angle
  -----------------------------------------------------------------

  local max_angles = config.max_angles; -- 8
  local annotDir = dataConfig.annotation.dir;

  local dataset = {};

  -- categories
  local categories = paths.dir(annotDir);
  RemoveDotDirs(categories);
  categories = removeExcludedCategories(categories, excluded_categories);
  
  local nClasses = table.getn(categories);

  for i=1,nClasses do
    dataset[categories[i]] = {};
  end

  for i=1,nClasses do

    -- videos
    local viddir = paths.concat(annotDir,categories[i]);
    local videos = paths.dir(viddir);
    RemoveDotDirs(videos);
    -- all viewpoint annotations will be similar to 00000_00's
    local angles = {};
    for k,v in pairs(videos) do
      local viewannot
      if paths.filep(paths.concat(annotDir, categories[i], videos[k], '00000_00_ge.mat')) then
        viewannot = mattorch.load(paths.concat(annotDir, categories[i], videos[k], '00000_00_ge.mat'));
      else
        viewannot = mattorch.load(paths.concat(annotDir, categories[i], videos[k], 'view.mat'));
      end
      -- if categories[i] == 'scenario6-basketball' then
      --   debugger.enter()
      -- end
      angles[k] = viewannot.ge;
    end

    for j=1,max_angles do --maximum 8 different angles
      dataset[categories[i]][j] = {};
    end

    for k,v in pairs(videos) do
      -- 1  018_03  scenario4-bowling
      -- if k == 1 and categories[i] == 'scenario4-bowling' then
      --   debugger.enter()
      -- end
      -- print(k,v,categories[i])
      dataset[categories[i]][angles[k][1][1]][v] = {};
    end

    for j=1,#dataset[categories[i]] do
      for k,v in pairs(dataset[categories[i]][j]) do
        local dir2  = paths.concat(annotDir,categories[i],k);
        local flist = paths.dir(dir2);
        RemoveDotDirs(flist);
        table.sort(flist, function (a,b) return a < b end);
        local pruned_flist = {}
        for id,f in pairs(flist) do
          if f:find("_00_ge.mat") then
            pruned_flist[#pruned_flist+1] = f
          end
        end
        dataset[categories[i]][j][k] = {};
        dataset[categories[i]][j][k] = pruned_flist;
      end
    end

  end

  dataset.config = dataConfig;
  return dataset;
end

function LoadTrainDatabase(exclude_category)
  return LoadDatabase(config.train, exclude_category)
end

function LoadTestDatabase(exclude_category)
  return LoadDatabase(config.test, exclude_category)
end

function ReadIndividualFrame(dataset, category, angle, video_id, imname, savefile, input_type)
  -----------------------------------------------------------------
  -- Reads a specific frame of a video for a category and an angle
  -- inputs:
  --       dataset:          The output of "LoadTrainDatabase"
  --       category:         Video category, e.g., 'sliding-ski', 'falling-diving', etc.
  --       angle:            View angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       video_id:         Video folder
  --       imname:           The name of frame's image to be read.
  --       savefile:         Save the tensor in this file.
  --       input_type:       The type of the data to be read. Should be one of
  --                         image, depth, normal or flow.
  -- output:
  --       images:  4D or 3D Tensor,
  --                [5 (orig + 4 crops) x] 3 (channels) x imH (image height) x imW (image width)
  -----------------------------------------------------------------
  local imH        = config.imH;
  local imW        = config.imW;
  local w_crop     = config.w_crop;

  local annotDir   = dataset.config.annotation.dir
  local trainDir   = dataset.config[input_type].dir;
  local image_type = dataset.config[input_type].type;
  local mean       = dataset.config[input_type].mean;
  local std        = dataset.config[input_type].std;

  local impath  = paths.concat(trainDir, category, video_id, imname .. "." .. image_type);
  local im     = loadImageOrig(impath);
  local imnorm = normalizeImage(image.scale(im, imW, imH), mean, std);

  local nChannels = dataset.config[input_type].nChannels;

  if w_crop and dataset.config[input_type].croppable then
    local images = torch.Tensor(5, nChannels, imH, imW)

    local coord  = mattorch.load(paths.concat(annotDir, category, video_id, imname .. "_00.mat"));
    local imSize = im:size();
    local height = imSize[2];
    local width  = imSize[3];

    local x1 = math.max(math.floor(coord.box[1][1]), 1);
    local y1 = math.max(math.floor(coord.box[1][2]), 1);
    local x2 = math.min(math.floor(coord.box[1][3]), width);
    local y2 = math.min(math.floor(coord.box[1][4]), height);

    local crop1 = im[{{},{y1,height},{x1,width}}];
    local crop2 = im[{{},{1,y2},{1,x2}}];
    local crop3 = im[{{},{y1,height},{1,x2}}];
    local crop4 = im[{{},{1,y2},{x1,width}}];
    images[1] = imnorm;
    images[2] = normalizeImage(image.scale(crop1, imW, imH), mean, std);
    images[3] = normalizeImage(image.scale(crop2, imW, imH), mean, std);
    images[4] = normalizeImage(image.scale(crop3, imW, imH), mean, std);
    images[5] = normalizeImage(image.scale(crop4, imW, imH), mean, std);

    for i=1,5 do
      images[i] = images[i][{{1,nChannels}, {}, {}}]
    end

    images = images:reshape(5 * nChannels, imH, imW)
    torch.save(savefile, images);
    return images
  else
    imnorm = imnorm[{{1, nChannels}, {}}]
    torch.save(savefile, imnorm);
    return imnorm
  end
end

function LoadIndividualFrame(dataset, category, angle, video_id, imname, input_type)
  -----------------------------------------------------------------
  -- Loads a specific frame of a video for a category and an angle
  -- inputs:
  --       dataset:          The output of "LoadTrainDatabase"
  --       category:         Video category, e.g., 'sliding-ski', 'falling-diving', etc.
  --       angle:            View angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       video_id:         Video folder
  --       imname:           The name of frame's image to be read.
  --       savefile:         Save the tensor in this file.
  --       input_type:       Optional type of the data to be read. Should be one of
  --                         image, depth, normal, flow or mask.
  -- output:
  --       images:  4D or 3D Tensor,
  --                [5 (orig + 4 crops) x] 3 (channels) x imH (image height) x imW (image width)
  -----------------------------------------------------------------
  if not input_type then
    local imH             = config.imH;
    local imW             = config.imW;
    local all_input_types = GetEnableInputTypes(dataset.config)
    local nChannels       = GetValuesSum(all_input_types)
    local result          = torch.Tensor(nChannels, imH, imW);

    local i = 1
    for input_type, nChannels in pairs(all_input_types) do
      result[{{i, i+nChannels-1}, {}, {}}] = LoadIndividualFrame(dataset, category, angle, video_id, imname, input_type)
      i = i + nChannels
    end
    return result
  end
  
  local suffix = dataset.config[input_type].suffix;
  local w_crop     = config.w_crop;

  local saveDir = dataset.config.save_dir;
  if not paths.dirp(saveDir) then
   paths.mkdir(saveDir)
  end
  -- NOTE: If we may have different oids for a video, we need to use different
  -- save paths for w_crop = true.
  local fname = paths.concat(saveDir, category .. '_' .. video_id .. '_' ..
    (w_crop and '1' or '0') .. '_' .. suffix .. '_' .. imname .. '.t7');

  if paths.filep(fname) then
    return torch.load(fname)
  else
    return ReadIndividualFrame(dataset, category, angle, video_id, imname, fname, input_type)
  end
end

function ReadTrainImagesPerVideo(dataset, category, angle, video_id, savefile, input_type)
  -----------------------------------------------------------------
  -- Reads training images for a video for a category and an angle
  -- inputs:
  --       dataset:          The output of "LoadTrainDatabase"
  --       category:         Video category, e.g., 'sliding-ski', 'falling-diving', etc.
  --       angle:            View angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       video_id:         Video folder
  --       savefile:        Save the tensor in this file.
  --       opts
  -- output:
  --       images:  5D Tensor,
  --                # of images x 5 (orig + 4 crops) x 3 (channels) x imH (image height) x imW (image width)
  -----------------------------------------------------------------

  local imH        = config.imH;
  local imW        = config.imW;

  local trainDir   = dataset.config[input_type].dir;
  local mean       = dataset.config[input_type].mean;
  local std        = dataset.config[input_type].std;
  local image_type = dataset.config[input_type].type;
  local w_crop     = config.w_crop;

  local nImages = #dataset[category][angle][video_id];

  local images
  if w_crop then -- FIXME(hessam): nChannel needs to be fixe
    images = torch.Tensor(nImages, 5, 3, imH, imW)
  else
    images = torch.Tensor(nImages, 3, imH, imW)
  end

  local cnt = 0;
  for _,f in ipairs(dataset[category][angle][video_id]) do
    cnt = cnt + 1;
    local matname = f;
    local imname, oid = f:match("([^_]+)_([^_]+)");

    images[cnt] = LoadIndividualFrame(dataset, category, angle, video_id, imname, input_type)
  end

  collectgarbage()
  torch.save(savefile, images)
  return images
end

function LoadTrainImagesPerVideo(dataset, category, angle, video_id, input_type)
  -----------------------------------------------------------------
  -- If files do not exist, it calls "ReadTrainImagesPerVideo" or "ReadTrainImagesPerVideoNoCrop".
  -- Otherwise, it loads from the disk.
  --
  -- inputs:
  --       dataset:          The output of "LoadTrainDatabase"
  --       category:         Video category, e.g., 'sliding-ski', 'falling-diving', etc.
  --       angle:            View angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       video_id:         Video folder
  --       opts
  -- outputs:
  --       images:           4D or 5D Tensor,
  --                         # of images x 5 (orig + 4 crops)? x 3 (channels) x
  --                         imH (image height) x imW (image width)
  -----------------------------------------------------------------

  local imH             = config.imH;
  local imW             = config.imW;
  local nImages = #dataset[category][angle][video_id];
  local images, nChannels
  if input_type then
    nChannels = dataset.config[input_type].nChannels
  else
    local all_input_types = GetEnableInputTypes(dataset.config)
    nChannels       = GetValuesSum(all_input_types)
  end
  images = torch.Tensor(nImages, nChannels, imH, imW)

  local cnt = 0;
  for _,f in ipairs(dataset[category][angle][video_id]) do
    cnt = cnt + 1;
    local matname = f;
    local imname, oid = f:match("([^_]+)_([^_]+)");

    images[cnt] = LoadIndividualFrame(dataset, category, angle, video_id, imname, input_type)
  end

  return images
end

function LoadRandomFrameOfVideo(dataset, category, angle, video_id, input_type)
  -----------------------------------------------------------------
  -- If files do not exist, it calls "ReadTrainImagesPerVideo" or "ReadTrainImagesPerVideoNoCrop".
  -- Otherwise, it loads from the disk.
  --
  -- inputs:
  --       dataset:          The output of "LoadTrainDatabase"
  --       category:         Video category, e.g., 'sliding-ski', 'falling-diving', etc.
  --       angle:            View angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       video_id:         Video folder
  --       opts
  -- outputs:
  --       images:           3D or 4D Tensor,
  --                         5 (orig + 4 crops)? x 3 (channels) x
  --                         imH (image height) x imW (image width)
  -----------------------------------------------------------------
  local randomFrame = GetRandomValue(dataset[category][angle][video_id])
  local imname = randomFrame:match('[^_]+')
  return LoadIndividualFrame(dataset, category, angle, video_id, imname, input_type)
end

