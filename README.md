# MOISTURE-MINDS

## Python Modules required

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import pickle
import warnings
```


## Input Data Format
This model accepts input data in the CSV file format. The input data should contain coloumns 
`pm1`, `pm2`,  `pm3` , `am`, `lum`, `st`,`sm `,`pres`, `humd`, `temp`.

## How to run
Run  `predict.py` python file with your input data csv file as command line argument. Make sure that `model.pkl` and `predict.py` are on same folder. You can use the following command when  `input_data.csv` is in the same folder as `predict.py`.

```
python predict.py input_data.csv
```

## Team Details

### Team Name : 642947-UY0H96J6
### Team Members :
* Tella Rajashekar Reddy
* Sahaja Nandyala
* Manche Pavanitha
* Kavali Sri Vyshnavi Devi
* Sripalle Meghana

