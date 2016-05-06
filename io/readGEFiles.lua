function LoadGEDatabase()
  -----------------------------------------------------------------
  -- Reads the list of images in the Game Engine videos
  -- outputs:
  --       dataset: a table with list of files for each category
  --                datatset[category][angle][variation], e.g.,
  --                datatset['rolling'][1]['1_1']
  -----------------------------------------------------------------
  os.execute('mkdir -p ' .. config.GE.saveDir)

  local geDir = config.GE.dir;

  local dataset = {};

  -- physics categories
  local physicsCategories = paths.dir(geDir);
  RemoveDotDirs(physicsCategories);

  local nClasses=table.getn(physicsCategories);

  for i=1,nClasses do
    dataset[physicsCategories[i]] = {};
  end

  for i=1,nClasses do

    -- angle directories
    local dir1 = paths.concat(geDir,physicsCategories[i]);
    local angleCategories = paths.dir(dir1);
    RemoveDotDirs(angleCategories);

    local anglebins = {}
    for _,a in pairs(angleCategories) do
      abin, tmp = a:match("([^,]+)_([^,]+)");
      anglebins[tonumber(abin)] = 1;
    end

    for k,_ in pairs(anglebins) do
      dataset[physicsCategories[i]][k] = {};
    end

    for k,_ in pairs(anglebins) do
      local dir2 = paths.concat(geDir,physicsCategories[i],tostring(k) .. '_' .. 'im');
      local alldirs = paths.dir(dir2);
      RemoveDotDirs(alldirs);
      table.sort(alldirs, function (a,b) return a < b end);

      for _,d in pairs(alldirs) do
        local dir3 = paths.concat(geDir,physicsCategories[i],tostring(k) .. '_' .. 'im', d);
        local files = paths.dir(dir3);
        RemoveDotDirs(files);

        table.sort(files, function (a,b) return a < b end);
        dataset[physicsCategories[i]][k][d] = {};
        dataset[physicsCategories[i]][k][d] = files;

      end

    end

  end

  local splines = mattorch.load(config.GE.splinesFile)['splines']
  setmetatable(dataset, {splines = splines})
  
  return dataset;
end


function ReadGEImagesPerCategory(physicsCategory, angle, dataset, savefile, input_type)
  -----------------------------------------------------------------
  -- Reads game engine videos for a category and an angle
  -- inputs:
  --       physicsCategory: 'rolling', 'falling', etc.
  --       angle:            the view angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       dataset:          the output of "LoadGEDatabase" function
  --       savefile:         the filename for stroing the videos in our format
  --       opts
  -- outputs:
  --       images:  5D Tensor,
  --                nvariations (different z's for the camera, different forces, etc. ) x
  --                fr_per_video x 3 (channels) x imH (image height) x imW (image width)
  -----------------------------------------------------------------

  local images  = {};

  local geDir = config.GE.dir;

  local imH          = config.GE.imH;
  local imW          = config.GE.imW;
  local fr_per_video = config.GE.frame_per_video; -- # of frames that we want to keep from each video

  local mean = config.GE[input_type].mean;
  local std = config.GE[input_type].std;
  local nChannels = config.GE[input_type].nChannels;

  local nvariations = getTableSize(dataset[physicsCategory][angle]);

  local imTensor     = torch.Tensor(nvariations, fr_per_video, nChannels, imH, imW);
  local suffix = config.GE[input_type].suffix

  local cnt = 0;
  for dir,files in pairs(dataset[physicsCategory][angle]) do
    cnt = cnt + 1;
    for f = 1,#files do
      local fname_im    = paths.concat(geDir,physicsCategory, angle .. '_' .. suffix, dir, files[f]);
      local im          = normalizeImage(image.scale(loadImageOrig(fname_im), imW, imH), mean, std);
      imTensor[cnt][f]  = im[{{1,nChannels}, {}, {}}];
    end
  end

  images  = imTensor;
  collectgarbage()
  torch.save(savefile, images)
  return images
end

function LoadGEPositionPerCategory(physicsCategory, angle, dataset)
  -----------------------------------------------------------------
  -- If files do not exist, it calls "ReadGEPositionPerCategory". Otherwise,
  -- it loads from the disk.
  --
  -- inputs:
  --       physicsCategory: 'rolling', 'falling', etc.
  --       angle:            the view angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       dataset:          the output of "LoadGEDatabase" function
  -- outputs:
  --       positions:  3D Tensor,
  --                nvariations (different z's for the camera, different forces, etc. ) x
  --                fr_per_video x 3 (x,y,z)
  -----------------------------------------------------------------

  saveDir = config.GE.saveDir;

  local positions;

  fname = paths.concat(saveDir, physicsCategory .. '_' .. angle .. '_positions' .. '.t7');

  if paths.filep(fname) then
    positions = torch.load(fname)
  else
    positions = ReadGEPositionPerCategory(physicsCategory, angle, dataset, fname);
  end

  return positions;
end



function LoadGEPerCategory(physicsCategory, angle, dataset, input_type)
  -----------------------------------------------------------------
  -- If files do not exist, it calls "ReadGEImagesPerCategory". Otherwise,
  -- it loads from the disk.
  --
  -- inputs:
  --       physicsCategory: 'rolling', 'falling', etc.
  --       angle:            the view angle (1 out of 8 or 1 out of 3 for symmetric categories)
  --       dataset:          the output of "LoadGEDatabase" function
  --       opts
  -- outputs:
  --       images:  5D Tensor,
  --                nvariations (different z's for the camera, different forces, etc. ) x
  --                fr_per_video x 3 (channels) x imH (image height) x imW (image width)
  -----------------------------------------------------------------
  if not input_type then
    local imH          = config.GE.imH;
    local imW          = config.GE.imW;
    local fr_per_video = config.GE.frame_per_video; -- # of frames that we want to keep from each video
    local nvariations = config.GE.use_multiple_vars and getTableSize(dataset[physicsCategory][angle]) or 1;
    local all_input_types = GetEnableInputTypes(config.GE)
    local nChannels = GetValuesSum(all_input_types)
    local result = torch.Tensor(nvariations, fr_per_video, nChannels, imH, imW);

    local i = 1
    for input_type, nChannels in pairs(all_input_types) do
      result[{{}, {}, {i, i+nChannels-1}, {}, {}}] = LoadGEPerCategory(physicsCategory, angle, dataset, input_type)
      i = i + nChannels
    end
    return result
  end


  local saveDir = config.GE.saveDir
  local suffix = config.GE[input_type].suffix

  local images;
  local fname = paths.concat(saveDir, physicsCategory .. '_' .. angle .. '_' .. suffix .. '.t7');

  if paths.filep(fname) then
    images = torch.load(fname)
  else
    images = ReadGEImagesPerCategory(physicsCategory, angle, dataset, fname, input_type);
  end

  if not config.GE.use_multiple_vars then
    local var_id = images:size(1) == 1 and 1 or 2
    images = images[{{var_id}, {}, {}, {}, {}}]
  end
  return images;
end


function ReadGEImagesAll(dataset, savefile, opts)
-----------------------------------------------------------------
  -- Reads all game engine videos
  -- inputs:
  --       dataset:          the output of "LoadGEDatabase" function
  --       savefile:         the filename for stroing the videos in our format
  --       opts
  -- outputs:
  --       images:  5D Tensor,
  --                nvariations (different z's for the camera, different forces, etc. ) x
  --                fr_per_video x 3 (channels) x imH (image height) x imW (image width)
  -----------------------------------------------------------------

  local images  = {};
  local geDir = config.GE.dir;

  local imH          = config.GE.imH;
  local imW          = config.GE.imW;
  local fr_per_video = config.GE.frame_per_video; -- # of frames that we want to keep from each video

  local input_type = opts.input_type;
  local mean = opts.mean;
  local std = opts.std;

  for physicsCategory,_ in pairs(dataset) do
    images[physicsCategory] = {};
    for angle,_ in pairs(dataset[physicsCategory]) do
      images[physicsCategory][angle] = {};

      local nvariations = getTableSize(dataset[physicsCategory][angle]);
      local imTensor     = torch.Tensor(nvariations, fr_per_video, 3, imH, imW);

      local cnt = 0;
      for dir,files in pairs(dataset[physicsCategory][angle]) do
        cnt = cnt + 1;
        for f = 1,#files do
          local fname_im     = paths.concat(geDir,physicsCategory, angle .. '_' .. input_type, dir, files[f]);
          local im     = normalizeImage(image.scale(loadImageOrig(fname_im), imW, imH), mean, std);
          imTensor[cnt][f] = im;
        end

      end
      images [physicsCategory][angle]  = imTensor;
    end
  end
  collectgarbage()

  torch.save(savefile, images)
  return images
end

function LoadGEAll(dataset, opts)
  -----------------------------------------------------------------
  -- If files do not exist, it calls "ReadGEImagesAll". Otherwise,
  -- it loads from the disk.
  --
  -- inputs:
  --       dataset: the output of "LoadGEDatabase" function
  --       opts
  -- outputs:
  --       images:  5D Tensor,
  --                nvariations (different z's for the camera, different forces, etc. ) x
  --                fr_per_video x 3 (channels) x imH (image height) x imW (image width)
  -----------------------------------------------------------------

  saveDir = config.GEsaveDir;
  local images;

  fname = paths.concat(saveDir, 'allGE_' .. opts.input_type .. '.t7');

  if paths.filep(fname) then
    images = torch.load(fname)
  else
    images = ReadGEImagesAll(dataset, fname, opts);
  end
  return images
end
