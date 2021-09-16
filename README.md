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

```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



choose x, y cols:

```python
x_cols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target_col = 'Species'
```

```python
n_targets = len(data[target_col].unique())
n_targets
```




    3



```python
X, y = prepare_data(data, x_cols=x_cols, target_col=target_col)
```

    x_cols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    target_col='Species'


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



    epoch =   0, val_loss = 2.080, accuracy = 0.37
    epoch =   1, val_loss = 1.909, accuracy = 0.37
    epoch =   2, val_loss = 1.770, accuracy = 0.53
    epoch =   3, val_loss = 1.655, accuracy = 0.70
    epoch =   4, val_loss = 1.561, accuracy = 0.70
    epoch =   5, val_loss = 1.482, accuracy = 0.73
    epoch =   6, val_loss = 1.416, accuracy = 0.77
    epoch =   7, val_loss = 1.360, accuracy = 0.77
    epoch =   8, val_loss = 1.311, accuracy = 0.80
    epoch =   9, val_loss = 1.269, accuracy = 0.83
    epoch =  10, val_loss = 1.233, accuracy = 0.87
    epoch =  11, val_loss = 1.200, accuracy = 0.87
    epoch =  12, val_loss = 1.171, accuracy = 0.87
    epoch =  13, val_loss = 1.145, accuracy = 0.87
    epoch =  14, val_loss = 1.121, accuracy = 0.87
    epoch =  15, val_loss = 1.099, accuracy = 0.90
    epoch =  16, val_loss = 1.079, accuracy = 0.90
    epoch =  17, val_loss = 1.061, accuracy = 0.90
    epoch =  18, val_loss = 1.044, accuracy = 0.90
    epoch =  19, val_loss = 1.028, accuracy = 0.93
    epoch =  20, val_loss = 1.013, accuracy = 0.93
    epoch =  21, val_loss = 0.999, accuracy = 0.93
    epoch =  22, val_loss = 0.986, accuracy = 0.97
    epoch =  23, val_loss = 0.973, accuracy = 1.00
    epoch =  24, val_loss = 0.961, accuracy = 1.00

