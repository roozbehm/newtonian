--[[
Input: a table of two inputs {M, k}, where
  M = an n-by-d matrix
  k = an m-by-d matrix
Output: a n-by-m matrix
Each element is an approximation of the cosine similarity between  a row in k and the 
corresponding row of M. It's an approximation since we add a constant to the
denominator of the cosine similarity function to remove the singularity when
one of the inputs is zero. 
--]]

local SmoothPairwiseCosineSimilarity, parent = torch.class('nn.SmoothPairwiseCosineSimilarity', 'nn.Module')

function SmoothPairwiseCosineSimilarity:__init(smoothen)
  parent.__init(self)
  self.gradInput = {}
  self.smooth = smoothen or 1e-5
end

function SmoothPairwiseCosineSimilarity:updateOutput(input)
  local M, k = unpack(input)
   assert(M:size(2)==k:size(2),"ERROR: dimensions are not equal !!!")
  self.rownorms = torch.cmul(M, M):sum(2):sqrt():view(M:size(1))
  self.colnorms = torch.cmul(k, k):sum(2):sqrt():view(k:size(1)) 
  self.rowcol = torch.ger(self.rownorms,self.colnorms);
  self.dot = M * (k:t());
  self.output:set(torch.cdiv(self.dot, self.rowcol + self.smooth))
  return self.output
end

function SmoothPairwiseCosineSimilarity:updateGradInput(input, gradOutput)
  local M, k = unpack(input)
  local nrow = M:size(1);
  local ncol = k:size(1);
  local ndim = k:size(2);
  
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  
  
  -- M gradient
  self.gradInput[1]:set(torch.cdiv(gradOutput, self.rowcol + self.smooth)*k)
    local scale = torch.cmul(self.output, (torch.repeatTensor(self.colnorms,nrow,1)))
      :cdiv(self.rowcol + self.smooth)
      :cmul(gradOutput):sum(2)
      :cdiv(self.rownorms+self.smooth)
    self.gradInput[1]:add(torch.cmul(-torch.repeatTensor(scale,1,ndim), M))

  -- k gradient
  self.gradInput[2]:set(torch.cdiv(gradOutput, self.rowcol + self.smooth):t()* M)
    local scale = torch.cmul(self.output, (torch.repeatTensor(self.rownorms,ncol,1):t()))
      :cdiv(self.rowcol + self.smooth)
      :cmul(gradOutput):sum(1)
      :cdiv(self.colnorms+self.smooth)
    self.gradInput[2]:add(torch.cmul(-torch.repeatTensor(scale,ndim,1):t(), k))

  
  return self.gradInput
end