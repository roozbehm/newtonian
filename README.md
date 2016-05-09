# N<sup>3</sup>: Newtonian Image Understanding: Unfolding the Dynamics of Objects in Statis Images
This is the source code for Newtonian Neural Networks N<sup>3</sup>, which predicts the dynamics of objects in scenes.

### Citation
If you find N<sup>3</sup> useful in your research, please consider citing:
```
@inproceedings{mottaghiCVPR16N3,
    Author = {Roozbeh Mottaghi and Hessam Bagherinezhad and Mohamamd Rastegari and Ali Farhadi},
    Title = {Newtonian Image Understanding: Unfolding the Dynamics of Objects in Static Images},
    Booktitle = {CVPR},
    Year = {2016}
}
```

### Requirements
This code is written in Lua, based on [Torch](http://torch.ch). If you are on [Ubuntu 14.04+](http://ubuntu.com), you can follow [this instruction](https://github.com/facebook/fbcunn/blob/master/INSTALL.md) to install torch.

You need the [VIND dataset](https://docs.google.com/forms/d/1OROeoj55hfhwiMsDuVyzMgfnhatTUOBGz0qGnMXor4Y/viewform). Extract it in the current directory, and rename it to `VIND`. Or you can put it somewhere else and change the `config.DataRootPath` in `setting_options.lua`.

### Training
To run the training:
```
th main.lua train
```

This trains the model on training data, and once in every 10 iterations, evalutates on one `val_images` batch. If you want to validate on `val_videos` go to `setting_options.lua` and change the line `valmeta = imvalmeta` to `valmeta = vidvalmeta`.

### Test
You need to [get the weights](https://drive.google.com/file/d/0B7H3g3rb2Blwcm51dXdKbGxzLTQ/view). Extract the weights in the current directory and rename it `weights`. To run the test:
```
th main.lua test
```

### License
This code is released under MIT License.
