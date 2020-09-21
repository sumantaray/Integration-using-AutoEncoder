# Integration of multi-omics data using AutoEncoder

Importing the python libraries

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
scaling the data
```
scale_data = pd.concat([scale_data, protein_data], axis=1, join='inner', sort=False)
max_value = scale_data.max().max()
min_value = scale_data.min().min()
print(max_value, min_value)
```
```
10.0 -1.84275189932292
```

```
src = FloatList(scale_data)
src = src.split_by_rand_pct()
src = src.label_from_lists(src.train, src.valid)
src = src.add_test(FloatList(scale_data))
src = src.databunch()
src.train_dl.batch_size=25  
x,y = next(iter(src.train_dl))
```

The autoencoder module

```
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder_rna = nn.Sequential(
            *bn_drop_lin(2000, 120, bn=False, actn=nn.LeakyReLU())
        )
        
        self.encoder_protein = nn.Sequential(
            *bn_drop_lin(10, 8, bn=False, actn=nn.LeakyReLU())
        )
        
        self.encoder_concat = nn.Sequential(
            *bn_drop_lin(128, 64, actn=nn.LeakyReLU())
        )
        
        self.decoder = nn.Sequential(
            *bn_drop_lin(64, 128, actn=nn.LeakyReLU()),
            *bn_drop_lin(128, 2010, actn=SigmoidRange(min_value, max_value))
        )


    def forward(self, x):
        x_rna = self.encoder_rna(x[:, :2000])
        x_protein = self.encoder_protein(x[:, 2000:])
        x = self.encoder_concat(torch.cat([x_rna, x_protein], 1))
        x = self.decoder(x)
        return x
        
 ```

Learn the Model
```
model = Autoencoder()
learn = Learner(src, model, loss_func=F.mse_loss)
learn.lr_find()
learn.fit_one_cycle(100, 0.06)
```
```
learn.model.encoder_concat
```

output:
```
Sequential(
  (0): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (1): Linear(in_features=128, out_features=64, bias=True)
  (2): LeakyReLU(negative_slope=0.01)
)
```

Get the representation from bottleneck layer
```
bottleneck = learn.model.encoder_concat[2]
hook = hook_output(bottleneck)
learn.data.test_dl.dl.sampler
bn_actns = []
for x, _ in learn.data.test_dl:
    learn.model(x)
    bn_actns.append(hook.stored)
```
Save the results

```
bn_actns = torch.cat(bn_actns)
bn_actns = bn_actns.cpu().numpy()
pd.DataFrame(bn_actns).to_csv("lowdim_rna_prot_separated.csv")
```

```
bn_actns.shape 
(7895, 64)
```




