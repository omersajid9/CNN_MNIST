from import_list import *
import scipy

class Data:
    def __init__(self, data_loc):
        self.original = np.array(pd.read_csv(data_loc), dtype=float)
        self.batch_data = self.original.copy()
        
    def shuffle_data(self):
        self.batch_data = self.original
        np.random.shuffle(self.batch_data)
        
    def divide_data(self, train_percentage):
        num_train = int(train_percentage * self.batch_data.shape[0])
        self.train_data = self.batch_data[:num_train, :]
        self.test_data = self.batch_data[self.train_data.shape[0]:, :]
        
    def y_to_vector(y_data):
        temp = np.zeros((y_data.shape[0], 10))
        temp[np.arange(y_data.shape[0]), y_data] = 1
        return temp

    def parse_data(self):
        self.y_train = np.array(self.train_data[:, 0], dtype=int)
        self.y_train_vector = Data.y_to_vector(self.y_train)
        self.x_train = self.train_data[:, 1:]  / 255.
        self.y_test = np.array(self.test_data[:, 0], dtype=int)
        self.y_test_vector = Data.y_to_vector(self.y_test)
        self.x_test = self.test_data[:, 1:]  / 255.

    def normalize_data(self):
        self.x_train -= np.mean(self.x_train, axis = 0)
        # self.x_train /= np.std(self.x_train, axis = 0)

    def rotate_data(self):
        M = self.x_train.shape[0]
        dim = int(np.sqrt(self.x_train.shape[1]))
        x = self.x_train.reshape(M, dim, dim)
        pos10 = scipy.ndimage.rotate(x, 5, axes=(1, 2), reshape=False).reshape(M, dim*dim)
        neg10 = scipy.ndimage.rotate(x, -5, axes=(1, 2), reshape=False).reshape(M, dim*dim)
        self.x_train = np.append(self.x_train, pos10, axis=0)
        self.x_train = np.append(self.x_train, neg10, axis=0)
        # print(self.y_train_vector.shape)
        temp = np.append(self.y_train_vector, self.y_train_vector, axis=0)
        self.y_train_vector = np.append(temp, self.y_train_vector, axis=0)
        # print(self.y_train_vector.shape)

        
    def prepare_data(self, train_percentage):
        self.shuffle_data()
        self.divide_data(train_percentage)
        self.parse_data()
        self.normalize_data()
        # print("Here")
        # if np.random.choice([0, 1]) == 1:
        # self.rotate_data()