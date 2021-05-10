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
from=203 # an implicit_add layer
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
from=207 # an implicit_mul layer
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
