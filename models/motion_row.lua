local C3D = nn.Sequential()

--------------------- Convolutional Layers ------------------
----------------------- 1st layer group ---------------------
C3D:add(cudnn.VolumetricConvolution(10,64,3,3,3,1,1,1,1,1,1))      -- Conv1a
C3D:add(cudnn.ReLU(true))
C3D:add(cudnn.VolumetricMaxPooling(1,2,2))
----------------------- 2nd layer group ---------------------
C3D:add(cudnn.VolumetricConvolution(64,64,3,3,3,1,1,1,1,1,1))    -- Conv2a
C3D:add(cudnn.ReLU(true))
C3D:add(cudnn.VolumetricMaxPooling(1,2,2))
----------------------- 3rd layer group ---------------------
C3D:add(cudnn.VolumetricConvolution(64,64,3,3,3,1,1,1,1,1,1))   -- Conv3a
C3D:add(cudnn.ReLU(true))
C3D:add(cudnn.VolumetricMaxPooling(1,2,2))
------------------------4th layer group-------------------------------
C3D:add(cudnn.VolumetricConvolution(64,64,3,3,3,1,1,1,1,1,1))   -- Conv3b
C3D:add(cudnn.ReLU(true))
C3D:add(cudnn.VolumetricMaxPooling(1,2,2))
----------------------- 5th layer group ---------------------
C3D:add(cudnn.VolumetricConvolution(64,64,3,3,3,1,1,1,1,1,1))   -- Conv4a
C3D:add(cudnn.ReLU(true))
C3D:add(cudnn.VolumetricMaxPooling(1,2,2))

C3D:add(nn.Max(3))
C3D:add(nn.View(64*8*8))     
C3D:add(cudnn.ReLU(true)) 
C3D:add(nn.Dropout(0.5))

return C3D
