# A simple pytorch based MNL lib
> Fit your Multinomial Logistic Regression with Pytorch


## Install

`pip install pytorch_mnl`

## How to use

import the lib

```python
import pandas as pd
from pytorch_mnl.core import *
```

load data

```python
data = pd.read_csv("./data/Iris.csv").drop("Id", axis=1)
```

choose x, y cols:

```python
x_cols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target_col = 'Species'
```

the number of classes to predict:

```python
n_targets = len(data[target_col].unique())
n_targets
```




    3



```python
X, y = prepare_data(data, x_cols=x_cols, target_col=target_col)
```

we get pytorch tensors ready to use!

```python
type(X), type(y)
```




    (torch.Tensor, torch.Tensor)



let's split in train/valid choosing a percenage as holdout, and choose a batch size to fit our model

```python
dls = DataLoaders.from_Xy(X, y, pct=0.2, batch_size=8)
```

as our model has 4 variables, we will fit a 4 MNL, with 3 targets.

```python
model = LinearMNL(len(x_cols), n_targets)
```

```python
learn = Learner(dls, model)
```

```python
learn.fit(25)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='25' class='' max='25' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [25/25 00:00<00:00]
</div>



    epoch =   0, val_loss = 2.411, accuracy = 0.17
    epoch =   1, val_loss = 2.223, accuracy = 0.27
    epoch =   2, val_loss = 2.031, accuracy = 0.50
    epoch =   3, val_loss = 1.873, accuracy = 0.63
    epoch =   4, val_loss = 1.743, accuracy = 0.70
    epoch =   5, val_loss = 1.636, accuracy = 0.70
    epoch =   6, val_loss = 1.546, accuracy = 0.73
    epoch =   7, val_loss = 1.471, accuracy = 0.77
    epoch =   8, val_loss = 1.408, accuracy = 0.77
    epoch =   9, val_loss = 1.353, accuracy = 0.77
    epoch =  10, val_loss = 1.306, accuracy = 0.80
    epoch =  11, val_loss = 1.264, accuracy = 0.83
    epoch =  12, val_loss = 1.227, accuracy = 0.83
    epoch =  13, val_loss = 1.195, accuracy = 0.83
    epoch =  14, val_loss = 1.165, accuracy = 0.87
    epoch =  15, val_loss = 1.139, accuracy = 0.87
    epoch =  16, val_loss = 1.115, accuracy = 0.87
    epoch =  17, val_loss = 1.093, accuracy = 0.87
    epoch =  18, val_loss = 1.072, accuracy = 0.87
    epoch =  19, val_loss = 1.054, accuracy = 0.93
    epoch =  20, val_loss = 1.036, accuracy = 0.97
    epoch =  21, val_loss = 1.020, accuracy = 0.97
    epoch =  22, val_loss = 1.005, accuracy = 0.97
    epoch =  23, val_loss = 0.991, accuracy = 0.97
    epoch =  24, val_loss = 0.977, accuracy = 0.97

