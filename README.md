# integration of multi-omics data using AutoEncoder

Importing th python libraries

```
import pandas as pd
from fastai import *
from fastai.tabular import * 
from fastai.callbacks import *
%matplotlib inline
```

Load the datasets 

```
#Load rna expression data
scale_data = pd.read_csv("rna_scaled.csv", index_col=0)
scale_data = scale_data.transpose()
max_value = scale_data.max().max()
min_value = scale_data.min().min()

#Load protein expression data

protein_data = pd.read_csv("protein_scaled.csv", index_col=0)
protein_data = protein_data.transpose()
max_value_p = protein_data.max().max()
min_value_p = protein_data.min().min()
```



