![Implicit Modeling](https://github.com/WongKinYiu/yolor/blob/main/figure/implicit_modeling.png)

### 1. silence layer

Usage:

```
[silence]
```

PyTorch code:

``` python
class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):    
        return x
```


### 2. implicit_add layer

Usage:

```
[implicit_add]
filters=128
```

PyTorch code:

``` python
class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit
```


### 3. shift_channels layer

Usage:

```
[shift_channels]
from=101
```

PyTorch code:

``` python
class ShiftChannel(nn.Module):
    def __init__(self, layers):
        super(ShiftChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) + x
```


### 4. implicit_mul layer

Usage:

```
[implicit_mul]
filters=128
```

PyTorch code:

``` python
class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit
```


### 5. control_channels layer

Usage:

```
[control_channels]
from=101
```

PyTorch code:

``` python
class ControlChannel(nn.Module):
    def __init__(self, layers):
        super(ControlChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) * x
```


### 6. implicit_cat layer

Usage:

```
[implicit_cat]
filters=128
```

PyTorch code: (same as ImplicitA)

``` python
class ImplicitC(nn.Module):
    def __init__(self, channel):
        super(ImplicitC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit
```


### 7. alternate_channels layer

Usage:

```
[alternate_channels]
from=101
```

PyTorch code:

``` python
class AlternateChannel(nn.Module):
    def __init__(self, layers):
        super(AlternateChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return torch.cat([a.expand_as(x), x], dim=1)
```


### 8. implicit_add_2d layer

Usage:

```
[implicit_add_2d]
filters=128
atoms=128
```

PyTorch code:

``` python
class Implicit2DA(nn.Module):
    def __init__(self, atom, channel):
        super(Implicit2DA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, atom, channel, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit
```


### 9. shift_channels_2d layer

Usage:

```
[shift_channels_2d]
from=101
```

PyTorch code:

``` python
class ShiftChannel2D(nn.Module):
    def __init__(self, layers):
        super(ShiftChannel2D, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1,-1,1,1)
        return a.expand_as(x) + x
```


### 10. implicit_mul_2d layer

Usage:

```
[implicit_mul_2d]
filters=128
atoms=128
```

PyTorch code:

``` python
class Implicit2DM(nn.Module):
    def __init__(self, atom, channel):
        super(Implicit2DM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, atom, channel, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit
```


### 11. control_channels_2d layer

Usage:

```
[control_channels_2d]
from=101
```

PyTorch code:

``` python
class ControlChannel2D(nn.Module):
    def __init__(self, layers):
        super(ControlChannel2D, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1,-1,1,1)
        return a.expand_as(x) * x
```


### 12. implicit_cat_2d layer

Usage:

```
[implicit_cat_2d]
filters=128
atoms=128
```

PyTorch code: (same as Implicit2DA)

``` python
class Implicit2DC(nn.Module):
    def __init__(self, atom, channel):
        super(Implicit2DC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, atom, channel, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit
```


### 13. alternate_channels_2d layer

Usage:

```
[alternate_channels_2d]
from=101
```

PyTorch code:

``` python
class AlternateChannel2D(nn.Module):
    def __init__(self, layers):
        super(AlternateChannel2D, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1,-1,1,1)
        return torch.cat([a.expand_as(x), x], dim=1)
```


### 14. dwt layer

Usage:

```
[dwt]
```

PyTorch code:

``` python
# https://github.com/fbcotter/pytorch_wavelets
from pytorch_wavelets import DWTForward, DWTInverse
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.xfm = DWTForward(J=1, wave='db1', mode='zero')

    def forward(self, x):
        b,c,w,h = x.shape
        yl, yh = self.xfm(x)
        return torch.cat([yl/2., yh[0].view(b,-1,w//2,h//2)/2.+.5], 1)
```
