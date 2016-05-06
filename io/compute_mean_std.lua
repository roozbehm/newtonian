function compute_mean_std(dataset, dataset_GE)
  --------------------  COMPUTE MEAN AND STD OF REAL VIDEOS -------------------
  for input_type, train_config in pairs(config.train) do
    if type(train_config) == 'table' and train_config.enable then
      local test_config = config.test[input_type]
      local meanstdFile = config.train.annotation.dir .. '/.meanstd_real_' .. input_type .. '.t7';
      if paths.filep(meanstdFile) then
        local meanstd = torch.load(meanstdFile)
        train_config.mean     = meanstd.mean;
        train_config.std    = meanstd.std;
        if test_config and test_config.enable then
          test_config.mean, test_config.std = train_config.mean, train_config.std;
        end
      else
        local trainDir = train_config.dir;
        local allfiles = MakeListTrainFrames(dataset, trainDir, train_config.type);
        train_config.mean, train_config.std = ComputeMeanStd(1000, allfiles, config.imH, config.imW);
        if test_config and test_config.enable then
          test_config.mean, test_config.std = train_config.mean, train_config.std;
        end
        local cache = {};
        cache.mean  = train_config.mean;
        cache.std   = train_config.std;
        torch.save(meanstdFile,cache);
      end
    end
  end


  -----------------  COMPUTE MEAN AND STD OF GAME ENGINE VIDEOS ----------------

  for input_type, conf in pairs(config.GE) do
    if type(conf) == 'table' and conf.enable then
      local meanstdFile = config.GE.dir .. '/.meanstd_GE_' .. input_type .. '.t7';
      if paths.filep(meanstdFile) then
        local meanstd = torch.load(meanstdFile)
        conf.mean    = meanstd.mean;
        conf.std     = meanstd.std;
      else
        local allfiles = MakeListGEFrames(dataset_GE, conf.suffix);
        conf.mean, conf.std = ComputeMeanStd(1000, allfiles, config.GE.imH, config.GE.imH);
        local cache = {};
        cache.mean  = conf.mean;
        cache.std   = conf.std;
        torch.save(meanstdFile,cache);
      end
    end
  end
end

function LoadCaffeMeanStd(meanFilePath)
  local meanFile = mattorch.load(meanFilePath)
  for input_type, train_config in pairs(config.train) do
    if type(train_config) == 'table' and train_config.enable then
      local test_config = config.test[input_type]
      for i=1,3 do
        train_config.mean[i] = meanFile.mean_data:select(3,i):mean() / 255
        train_config.std[i]  = 1/255
      end
      if test_config and test_config.enable then
        test_config.mean, test_config.std = train_config.mean, train_config.std;
      end
    end
  end
end
