from import_list import *

class LeNet5:
    def __init__(self):
        self.cost_list = list()
        self.loss_list = list()
        self.accuracy_list = list()
        


        self.layout = list()
        k1, Cin1, Cout1, k2, Cin2, Cout2, nodes1, nodes2, nodes3, learning_rate = 3, 1, 8, 3, 8, 16, 1352, 200, 10, .01
        self.layout.append(Conv(3, 1, 16, padding=2, learning_rate=0.007))
        self.layout.append(BatchNorm(16, 0.007, dim=2))
        self.layout.append(ReLU())
        self.layout.append(Maxpool())
        # self.layout.append(BatchNorm(16, 0.007, dim=2))
        # self.layout.append(Conv(3, 16, 32, learning_rate=0.007))
        # self.layout.append(BatchNorm(32 , 0.007, dim=2))

        # self.layout.append(ReLU())
        # self.layout.append(Maxpool())
        # self.layout.append(Dropout(0.1))
        # self.layout.append(Conv(1, 32, 32, padding=0, learning_rate=0.007))
        # self.layout.append(ReLU())
        # self.layout.append(BatchNorm(32, 0.007, dim=2))

        # self.layout.append(Maxpool())
        self.layout.append(Flatten())
        self.layout.append(FC(3600, 128, 0.007))
        self.layout.append(BatchNorm(128, 0.007))
        self.layout.append(ReLU())
        self.layout.append(FC(128, 128, 0.007))
        self.layout.append(ReLU())
        self.layout.append(FC(128, 10, 0.007))
        self.layout.append(softmax())

    def forward_feed(self, input):
        for index, i in enumerate(self.layout):
            # print(index, np.mean(input))
            input = i.forward_propogation(input)
            # print(i.input["data"].shape)
        self.prediction = input
        
    def backward_feed(self, actual):
        self.actual = actual
        for j in self.layout[::-1]:
            actual = j.backward_propogation(actual)

    def update_params(self):
        for index, i in enumerate(self.layout):
            i.update_params()

    def batch_accomodating_forward(self, test_image):
        for i in self.layout:
            if isinstance(i, BatchNorm) :
                test_image = i.forward_propogation(test_image, mode="test")
            else:
                test_image = i.forward_propogation(test_image)
        self.prediction = test_image

    def predict(self, test_image):
        for i in self.layout:
            if isinstance(i, BatchNorm):
                print("Norm")
                test_image = i.forward_propogation(test_image, mode="test")
            else:
                test_image = i.forward_propogation(test_image)
        # self.forward_feed(test_image)
        output = np.argmax(self.layout[-1].output["data"])
        print("prediction")
        print(output)
        print(self.layout[-1].output["data"][0][output])
        return (output, self.layout[-1].output["data"][0][output])

    def get_accuracy(self, images, label):
        M = images.shape[0]
        dim = int(np.sqrt(images.shape[1]))
        images = images.reshape(M, 1, dim, dim)
        self.batch_accomodating_forward(images)
        pred = self.layout[-1].output["data"]
        return sum(np.argmax(pred, axis=1) == np.argmax(label, axis=1)) / pred.shape[0]



    def accuracy(self):
        return sum(np.argmax(self.prediction, axis=1) == np.argmax(self.actual, axis=1)) / self.prediction.shape[0]

    def compute_cost(self):
        act = self.actual.T
        pred = self.prediction.T
        m = self.actual.shape[0]

        cost_sum = np.sum((pred * act)) / m
        cost = - np.log(cost_sum)
        return cost

    def L_i_vectorized(self):
        act = self.actual.T
        pred = self.prediction.T
        delta = 1.0
        margins = np.maximum(0, pred - pred[act==1] + delta)
        margins[act==1] = 0
        loss_i = np.sum(margins)
        return loss_i/self.prediction.shape[0]

    def printTestStatement(self, iteration, epoch, images, actual):
        M = images.shape[0]
        dim = int(np.sqrt(images.shape[1]))
        images = images.reshape(M, 1, dim, dim)
        self.batch_accomodating_forward(images)
        self.actual = actual
        return self.printStatement(iteration, epoch)

    def printStatement(self, iteration, epoch):
        self.loss_list.append(self.L_i_vectorized())
        self.cost_list.append(self.compute_cost())
        self.accuracy_list.append(self.accuracy())
        statement = "Loss :" + str(self.L_i_vectorized()) + "; Accuracy: " + str(self.accuracy()) + "; Cost :" + str(self.compute_cost()) +"; " + str(iteration) + " / " + str(epoch)
        return statement

    def plot(self):
        plt.plot(range(len(self.loss_list)), np.log(self.loss_list), "red", label="Loss (Min " + str(np.min(self.loss_list)) + ")")
        plt.plot(range(len(self.cost_list)), self.cost_list, "blue", label="Cost (Min " + str(np.min(self.cost_list)) + ")")
        plt.plot(range(len(self.accuracy_list)), self.accuracy_list, "green", label="Accuracy (Max " + str(np.max(self.accuracy_list)) + ")")
        plt.legend()
