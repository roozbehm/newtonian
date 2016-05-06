---- options
config={};

config.GPU  = 1
config.nGPU = 1

config.DataRootPath = "data"
config.SaveRootPath = "data/logs"
config.CacheRootPath = "cache"

config.logDirectory = config.SaveRootPath .. '/' .. "LOG_" .. os.getenv('USER') .. "_" .. os.date():gsub(' ','-');
os.execute('mkdir -p ' .. config.logDirectory)
config.logFile = assert(io.open(paths.concat(config.logDirectory, 'log.txt'), 'w'))

config.GE = {
  image = {
    suffix = 'im',
    mean = {},
    std = {},
    nChannels = 3,
    enable = true,
  },
  depth = {
    suffix = 'depth',
    mean = {},
    std = {},
    nChannels = 1,
    enable = true,
  },
  normal = {
    suffix = 'normal',
    mean = {},
    std = {},
    nChannels = 3,
    enable = true,
  },
  flow = {
    suffix = 'flow',
    mean = {},
    std = {},
    nChannels = 3,
    enable = true,
  },
  imH               = 256,
  imW               = 256,
  frame_per_video   = 10,
  use_multiple_vars = false,
  dir               = config.DataRootPath .. "/ge_videos",
  saveDir           = config.CacheRootPath .. "/ge_cache",
  featsDir          = config.CacheRootPath .. "/ge_feats",
  splinesFile       = config.DataRootPath .. "/ge_videos/.splines.mat",
}

config.imH = 227;
config.imW = 227;
config.max_angles = 8;

config.train = {
  annotation = {
    dir = config.DataRootPath .. "/train/labels",
  },
  image = {
    dir       = config.DataRootPath .. "/train/images",
    nChannels = 3,
    type      = "png",
    suffix    = "im",
    mean      = {},
    std       = {},
    enable    = true,
    croppable = true,
  },
  depth = {
    enable = false,
  },
  normal = {
    enable = false,
  },
  flow = {
    enable = false,
  },
  mask = {
    dir       = config.DataRootPath .. "/train/objmask",
    nChannels = 1,
    type      = "png",
    suffix    = "mask",
    mean      = {},
    std       = {},
    enable    = true,
  },
  save_dir = config.CacheRootPath .. "/train_cache",
  batch_size = 128,
  nIter = 1000000,
}

valmeta = {
  annotation = {
    dir = config.DataRootPath .. "/val/labels",
  },
  image = {
    dir       = config.DataRootPath .. "/val/images",
    nChannels = 3,
    type      = "png",
    suffix    = "im",
    mean      = {},
    std       = {},
    enable    = true,
    croppable = true,
  },
  depth = {
    enable = false,
  },
  normal = {
    enable = false,
  },
  flow = {
    enable = false,
  },
  mask = {
    dir       = config.DataRootPath .. "/val/objmask",
    nChannels = 1,
    type      = "png",
    suffix    = "mask",
    mean      = {},
    std       = {},
    enable    = true,
  },
  save_dir = config.CacheRootPath .. "/val_cache",
  batch_size = 243,
  nIter = 6,
}

testmeta = {
  annotation = {
    dir = config.DataRootPath .. "/test/labels",
  },
  image = {
    dir       = config.DataRootPath .. "/test/images",
    nChannels = 3,
    type      = "png",
    suffix    = "im",
    mean      = {},
    std       = {},
    enable    = true,
    croppable = true,
  },
  depth = {
    enable = false,
  },
  normal = {
    enable = false,
  },
  flow = {
    enable = false,
  },
  mask = {
    dir       = config.DataRootPath .. "/test/objmask",
    nChannels = 1,
    type      = "png",  
    suffix    = "mask",
    mean      = {},
    std       = {},
    enable    = true,
  },
  save_dir = config.CacheRootPath .. "/test_cache",
  batch_size = 3,
  nIter = 1,
}

config.test = mode == 'train' and valmeta or testmeta
config.classes = {'scenario11', 'scenario3', 'scenario10', 'scenario7', 'scenario6', 'scenario12', 'scenario9', 'scenario4', 'scenario1', 'scenario2', 'scenario8', 'scenario5'}
config.class_angles= {3, 8, 8, 3, 3, 4, 8, 8, 8, 4, 8, 1};

-- excluded_categories is a list of regexes. In lua, to escape special chars you need to add %
config.excluded_categories = {};

--------   BEGIN: Network configuration  -----
if mode == 'test' then
  config.nIter    = config.test.nIter
  config.batchSize = config.test.batch_size
else
  config.nIter    = config.train.nIter
  config.batchSize = config.train.batch_size
end
  
config.nDisplay = 1;
config.saveModelIter = 500;
config.nResetLR = 50000;
config.nCategories = 66
config.nClasses = config.nCategories
config.nEval    = 10;
config.lambda   = 0.5

config.initModelPath = {  imageNN = "caffe"
                        , animNN = "weights/motion_row.t7"
                        , fullNN = "weights/N3.t7" }

config.caffeInit = true;
config.caffeFilePath = {  
  proto  = 'weights/zoo/deploy.prototxt',
  model  = 'weights/zoo/bvlc_alexnet.caffemodel',
  mean   = 'weights/zoo/ilsvrc_2012_mean.mat'
};
config.regimes = {
    -- start, end,    LR,
    {  1,     100,   1e-2, },
    { 101,     1000,   1e-2, },
    { 1001,     10000,   1e-3, },
    {10001,      100000,  1e-4,},
  };
config.dropoutProb = 0.5;

--------   END :  Network configuration -------
