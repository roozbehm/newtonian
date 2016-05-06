log('Loading Train Functions ... ')

function train()
  config.testing = false

  local batchSize = config.batchSize;
  local animFeatures = GetAnimationFeatures(model.animationNN);

  for iter=1,config.nIter do
    ---- load one batch
    tt = iter
    local tic= os.clock()
    local imgFeatures, TrTarget = GetAUniformImageBatch(batchSize, {
                              viewpoint = true,
                              test      = false,
                              spline    = false,
                              })
    local TrInput = {imgFeatures,animFeatures};
    local toc = os.clock() - tic;
    log('loading time :' .. tostring(toc))
    
    -------- train the network--------------
    model.learningRate = model:LearningRateComp(iter);
    local acc, loss = model:TrainOneBatch(TrInput,TrTarget);
    if (iter % 10) == 0 then
      local  tic = os.clock()
      collectgarbage();
      local toc = os.clock() - tic;
      print("garbage collection :", toc)
    end
    if (iter % config.nDisplay) == 0 then
      log(('Iter = %d | Train Accuracy = %f | Train Loss = %f\n'):format(iter,acc,loss));
    end

    if (iter % config.nEval) == 0 then
      local TeInput, TeTarget = GetAUniformImageBatch(batchSize, {
                                  viewpoint = true,
                                  test      = true,
                                  spline    = false,
                                  });      
      local acc, loss = model:EvaluateOneBatch(TeInput,TeTarget);
      log(('Testing ---------> Iter = %d | Test Accuracy = %f | Test Loss = %f\n'):format(iter,acc,loss));
    end
    
    if (iter % config.saveModelIter) == 0 then
      local fileName = 'Model_iter_' .. iter .. '.t7';
      log('Saving NN model in ----> ' .. paths.concat(config.logDirectory, fileName) .. '\n');
      model:SaveModel(paths.concat(config.logDirectory, fileName));
    end

  end
end


---------------------------------------------------------
function test()
  config.testing = true
  ----------------------------

  local batchSize = config.batchSize;
  local meanAcc = 0;
  local sumFrameAcc = 0;
  local sumFramables = 0;
  local per_class_cum = torch.Tensor(config.nCategories, 2):fill(0)
  local all_predictions

  for iter=1,config.nIter do
    tt = iter
    ---- load one batch
    local tic= os.clock()
    local TeInput, TeTarget = GetAnImageBatch(batchSize, {
                                        viewpoint     = true,
                                        test          = true,
                                        deterministic = true,
                                        spline        = false,
                                      });
    local toc = os.clock() - tic;
    log('loading time :' .. tostring(toc))
    
    if (iter % 10) == 0 then
      local  tic = os.clock()
        collectgarbage();
      local toc = os.clock() - tic;
      print("garbage collection :", toc)
    end
    local acc, loss, per_class, predicts, frames = model:EvaluateOneBatch(TeInput,TeTarget);
    meanAcc = ((iter -1)* meanAcc + acc)/ iter;
    per_class_cum = per_class_cum + per_class
    
    log(('Iter = %d | Current Test Accuracy = %f | Average Test Accuracy = %f\n'):format(iter,acc,meanAcc));

    local predictions = torch.cat(TeTarget, predicts, 2)
    if not all_predictions then
      all_predictions = predictions
    else
      all_predictions = torch.cat(all_predictions, predictions, 1)
    end
  end
end
