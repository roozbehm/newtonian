paths.dofile('io/readFiles.lua')
paths.dofile('io/readGEFiles.lua');
paths.dofile('io/compute_mean_std.lua');
paths.dofile('io/readBatch.lua');
function GetASiameseBatch(nPositiveImages, nDifferentAngleImages, nDifferentCategoryImages, opt)
  local status, input, target = coroutine.resume(GetASiameseBatchCoroutine,
              nPositiveImages, nDifferentAngleImages, nDifferentCategoryImages, opt.test);
  return input, target
end

function GetAnImageBatch(batchSize, opt)
  local status, input, target = coroutine.resume(GetAnImageBatchCoroutine,
                                                 batchSize, opt.viewpoint, opt.test,
                                                 opt.deterministic, opt.spline);
  return input, target
end

function GetAUniformImageBatch(batchSize, opt)
  local status, input, target = coroutine.resume(GetAUniformImageBatchCoroutine,
                                                 batchSize, opt.viewpoint, opt.test,
                                                 opt.spline);
  return input, target
end

function GetAUniformAnimationBatch(batchSize, opt)
  local status, input, target = coroutine.resume(GetAUniformAnimationBatchCoroutine,
                                                 batchSize, opt.viewpoint, opt.spline);
  return input, target
end

function GetAVideoBatch(opt)
  local status, input, target = coroutine.resume(GetAVideoBatchCoroutine,
                                                 opt.viewpoint, opt.test, opt.spline);
  return input, target
end
