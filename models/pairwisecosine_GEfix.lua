local mlp = nn.Sequential();

mlp:add(nn.SmoothPairwiseCosineSimilarity());
mlp:add(nn.Reshape(config.batchSize,config.nClasses,10,false));
mlp:add(nn.Exp());
mlp:add(nn.Sum(3));

mlp:add(nn:LogSoftMax())
return mlp;