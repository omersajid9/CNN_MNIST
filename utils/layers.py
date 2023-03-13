from import_list import *

class Conv:
    def __init__(self, filter_dim, Cin, Cout, stride = 1, padding = 0, learning_rate = 0.001):
        self.filter_dim = filter_dim
        self.Cin = Cin
        self.Cout = Cout
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate
        self.W  = {"data": np.random.normal(0.0, np.sqrt(1/Cin*filter_dim*filter_dim), (Cout, Cin, filter_dim, filter_dim)), "delta": None}
        self.b = {"data": np.random.randn((self.Cout)), "delta": None}

        self.input = {"data": None, "delta": None}
        self.output = {"data": None, "delta": None}

    def forward_propogation(self, input):
        self.input["data"] = np.pad(input, ( (0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        M, Cin, h, _ = self.input["data"].shape
        new_h = int((h - self.filter_dim) / self.stride + 1)
        output = np.zeros((M, self.Cout, new_h, new_h))
        for i in range(0, new_h, self.stride):
            for j in range(0, new_h, self.stride):
                output[:, :, i, j] = (self.input["data"][:, np.newaxis, :, i : (i + self.filter_dim), j : (j + self.filter_dim)] * self.W["data"][np.newaxis, :, :, :, :]).sum((2, 3, 4)) + self.b["data"]
        self.output["data"] = output
        return self.output["data"]

    def zero_padding(self, inputs, size):
        h, _ = inputs.shape
        new_h = 2 * size + h
        output = np.zeros((new_h, new_h))
        output[size:h+size, size:h+size] = inputs
        return output

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        M, Cin, H, _ = self.input["data"].shape
        dW = np.zeros((self.W["data"].shape))
        dX = np.zeros((self.input["data"].shape))

        for i in range(self.filter_dim):
            for j in range(self.filter_dim):
                dW[:, :, i, j] = (self.input["data"][:, np.newaxis, :, i:(i + doutput.shape[-1]), j:(j + doutput.shape[-1])] * doutput[:, :, np.newaxis, :, :]).sum(axis=(0, 3, 4)) / M
        
        pad = self.filter_dim - 1
        n = np.pad(doutput, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        rot_w = np.rot90(np.rot90(self.W["data"], axes=(2, 3)), axes=(2, 3))
        for i in range(doutput.shape[-1]):
            for j in range(doutput.shape[-2]):
                dX[:, :, i + pad, j + pad] = (n[:, :, np.newaxis, i:(i + self.filter_dim), j:(j + self.filter_dim)] * rot_w[np.newaxis, :, :, :, :]).sum(axis=(1, 3, 4))
        self.W["delta"] = dW
        self.input["delta"] = dX
        self.b["delta"] = np.sum(doutput, axis=(0, 2, 3)) / M
        return self.input["delta"]


    def update_params(self):
        self.W["data"] = self.W["data"] - np.multiply(self.W["delta"], self.learning_rate)
        self.b["data"] = self.b["data"] - np.multiply(self.b["delta"], self.learning_rate)
        

class Maxpool:
    def __init__(self, factor=2):
        self.factor = factor

        self.input = {"data": None, "delta": None}
        self.output = {"data": None, "delta": None}

    def forward_propogation(self, input):
        self.input["data"] = input
        minus_inf = np.full((input.shape[0], input.shape[1], input.shape[2] // self.factor, input.shape[3] // self.factor), -float('inf'), dtype=input.dtype)
        np.maximum.at(minus_inf, (np.arange(input.shape[0])[:, None, None, None], np.arange(input.shape[1])[:, None, None], np.arange(input.shape[2])[:, None] // self.factor, np.arange(input.shape[3]) // self.factor), input)
        self.output["data"] = minus_inf
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        self.input["delta"] = np.repeat(np.repeat(self.output["delta"], self.factor, axis=2), self.factor, axis=3)
        return self.input["delta"]

    def update_params(self):
        pass


class Flatten:
    def __init__(self):
        self.input = {"data": None, "delta": None}
        self.output = {"data": None, "delta": None}

    def forward_propogation(self, input):
        self.input["data"] = input
        self.N, self.Cout, self.H, _ = input.shape
        self.output["data"] = input.reshape(input.shape[0], np.prod(input.shape[1:]))
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        self.input["delta"] = doutput.reshape(self.N, self.Cout, self.H, self.H)
        return self.input["delta"]

    def update_params(self):
        pass
    
class FC:
    def __init__ (self, num_in, num_out, learning_rate):
        self.learning_rate = learning_rate
        self.num_in = num_in
        self.num_out = num_out
        self.W = {"data": np.random.randn(num_in, num_out) / np.sqrt(num_in / 2), "delta":0}
        self.b = {"data": np.random.normal(loc = 0.0, scale = 0.5, size = (1, num_out)) / np.sqrt(num_out) , "delta": 0}

        self.input = {"data":None, "delta":None}
        self.output = {"data":None, "delta":None}

    def forward_propogation(self, input):
        self.input["data"] = input
        self.output["data"] = np.dot(self.input["data"], self.W["data"]) + self.b["data"]
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        self.W["delta"] = np.dot(self.input["data"].T, self.output["delta"]) / doutput.shape[0] 
        self.b["delta"] = np.sum(self.output["delta"], keepdims=True, axis=(0)) / doutput.shape[0] 
        self.input["delta"] = np.dot(self.output["delta"], self.W["data"].T) 
        return self.input["delta"]
    
    def update_params(self):
        self.W["data"] = self.W["data"] - np.multiply(self.W["delta"], self.learning_rate)
        self.b["data"] = self.b["data"] - np.multiply(self.b["delta"], self.learning_rate)

        # self.W["delta"] = None
        # self.b["delta"] = None

class ReLU:
    def __init__(self):
        self.input = {"data":None, "delta":None}
        self.output = {"data":None, "delta":None}
        

    def forward_propogation(self, input):
        self.input["data"] = input
        self.output["data"] = np.maximum(input, 0)
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        self.input["delta"] = self.output["delta"].copy()
        self.input["delta"][ self.input["data"] < 0 ] = 0
        return self.input["delta"]

    def update_params(self):
        pass

class tanh:
    def __init__(self):
        self.input = {"data":None, "delta":None}
        self.output = {"data":None, "delta":None}

    def forward_propogation(self, input):
        self.input["data"] = input
        self.output["data"] = np.tanh(input)
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        self.input["delta"] = doutput * (1 - self.output["data"]**2)
        return self.input["delta"]

    def update_params(self):
        pass

class sigmoid:
    def __init__(self):
        self.input = {"data":None, "delta":None}
        self.output = {"data":None, "delta":None}

    def forward_propogation(self, input):
        self.input["data"] = input
        self.output["data"] = 1 / (1 + np.exp( - input))
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        self.input["delta"] = np.exp( - doutput) / ((np.exp(- doutput) + 1) **2)
        return self.input["delta"]


class softmax:
    def __init__(self):
        self.input = {"data":None, "delta":None}
        self.output = {"data":None, "delta":None}

    def forward_propogation(self, input):
        self.input["data"] = input 
        # print(self.input["data"][0])
        self.output["data"] = np.exp(input) / np.sum(np.exp(input), 1, keepdims=True)
        
        return self.output["data"]

    def backward_propogation(self, dy):
        self.output["delta"] = dy
        self.input["delta"] = self.output["data"] - self.output["delta"]
        return self.input["delta"]

    def update_params(self):
        pass

class Dropout:
    def __init__(self, prob = 0.5):
        self.prob = prob

        if prob != 1.:
            self.scale = 1. / (1. - prob)
        else:
            self.scale = 1.  

        self.input = {"data":None, "delta":None}
        self.output = {"data":None, "delta":None}

    def forward_propogation(self, input):
        self.input["data"] = input
        self.mask = np.random.uniform(low=0., high=1., size=input.shape) >= self.prob
        self.output["data"] = self.mask * input * self.scale
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        self.input["delta"] =doutput * self.mask * self.scale
        return self.input["delta"]

    def update_params(self):
        pass


class BatchNorm:
    def __init__(self, num_in, learning_rate, dim = 1):
        self.input = {"data":None, "delta":None}
        self.output = {"data":None, "delta":None}

        if dim == 1:
            shape = (1, num_in)
            self.axi = (0)
        else:
            shape = (1, num_in, 1, 1)
            self.axi = (0, 2, 3)

        self.gamma = {"data": np.ones(shape, dtype=np.float64), "delta": 0}
        self.bias = {"data": np.zeros(shape, dtype=np.float64), "delta": 0}
        self.running = {"mean": 0, "var": 0, "gamma": 0.9}
        self.eps = 1e-5

        self.var = {"data": 0 , "delta": 0}
        self.mean = {"data": 0, "delta": 0}
        self.stddev = {"data": 0, "delta": 0}
        self.x_minus_mean = {"data": 0, "delta": 0}
        self.standard = {"data": 0, "delta": 0}
        self.learning_rate = learning_rate


    def update_running_variables(self):
        is_mean_empty = np.array_equal(np.zeros(0), self.running["mean"])
        is_var_empty = np.array_equal(np.zeros(0), self.running["var"])

        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Variance should be initialized at same time")
        if is_mean_empty:
            self.running["mean"] = self.mean["data"].copy()
            self.running["var"] = self.var["data"].copy()
        else:
            gamma = self.running["gamma"]
            self.running["mean"] = gamma * self.running["mean"] + (1 - gamma) * self.mean["data"]
            self.running["var"] = gamma * self.running["var"] + (1 - gamma) * self.var["data"]
            # print(self.running["mean"].shape)
            # print(self.running["mean"][0][10])
            # print(self.running["var"][0][10])
            # print(self.var["data"][0][10])

    def forward_propogation(self, input, mode="train"):
        self.mode = mode
        self.input["data"] = input
        M = input.shape[0]

        if mode == "test":
            self.mean["data"] = self.running["mean"].copy()
            self.var["data"] = self.running["var"].copy()

        else:
            assert len(input.shape) in (2, 4)
            self.mean["data"] = np.mean(input, axis=self.axi, keepdims=True)
            self.var["data"] = np.var(input, axis=self.axi, keepdims=True)
            self.update_running_variables()
        

        self.var["data"] += self.eps
        self.stddev["data"] = np.sqrt(self.var["data"])
        self.x_minus_mean["data"] = input - self.mean["data"]
        self.standard["data"] = self.x_minus_mean["data"] / self.stddev["data"]
        self.output["data"] = self.gamma["data"] * self.standard["data"] + self.bias["data"]
        return self.output["data"]

    def backward_propogation(self, doutput):
        self.output["delta"] = doutput
        M = doutput.shape[0]
        self.standard["delta"] = doutput * self.gamma["data"]
        self.var["delta"] = np.sum(self.standard["delta"] * self.x_minus_mean["data"] * -0.5 * self.var["data"] ** (-3/2), axis = self.axi, keepdims=True)
        self.stddev["delta"] = 1/self.stddev["data"]
        self.x_minus_mean["delta"] = 2 * self.x_minus_mean["data"] / M
        self.mean["delta"] = np.sum(self.standard["delta"] * - self.stddev["delta"], axis=self.axi, keepdims=True) + self.var["delta"] * np.sum(- self.x_minus_mean["delta"], axis=self.axi, keepdims=True)
        self.gamma["delta"] = np.sum(doutput * self.standard["data"], axis=self.axi, keepdims=True)
        self.bias["delta"] = np.sum(doutput, axis=self.axi, keepdims=True)
        self.input["delta"] = self.standard["delta"] * self.stddev["delta"] + self.var["delta"] * self.x_minus_mean["delta"] + self.mean["delta"] / M
        return self.input["delta"]

    def update_params(self):
        self.gamma["data"] -= self.learning_rate * self.gamma["delta"]
        self.bias["data"] -= self.learning_rate * self.bias["delta"]
        
