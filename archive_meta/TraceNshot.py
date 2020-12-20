import  torchvision.transforms as transforms
import  os.path
import  numpy as np
from numpy import newaxis

class TraceNshot:

    def __init__(self, data_train, data_test, label_train, label_test, batchsz, n_way, k_shot, k_query, num_instance = 199998):
        """
        :param data: 
        :param label: 
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        """

        self.x_train = data_train
        self.x_test = data_test
        self.y_train = label_train
        self.y_test = label_test

        '''
        IMPORTANT HERE!!!
        11/15 Ta-Yang change this to all versus all
        '''

        ### Original Setting
        # self.x_train, self.x_test = self.x[:num_train], self.x[num_train:]
        # self.y_train, self.y_test = self.y[:num_train], self.y[num_train:]


        ### Train ALL test ONE
        # self.x_train, self.x_test = self.x, self.x[trace_id]
        # self.y_train, self.y_test = self.y, self.y[trace_id]

        ### If given 2D then change to 3D

        if self.x_train.ndim < 3:
            self.x_train = self.x_train[newaxis, ...]

        if self.x_test.ndim < 3:
            self.x_test = self.x_test[newaxis, ...]

        if self.y_train.ndim < 3:
            self.y_train = self.y_train[newaxis, ...]

        if self.y_test.ndim < 3:
            self.y_test = self.y_test[newaxis, ...]       
             
        # self.x_test = self.x_test[newaxis, ...]
        # self.y_test = self.y_test[newaxis, ...]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = 1 #self.x_train.shape[0]=13  # 13
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.num_instance = num_instance
        self.window = 3
        self.bit_size = 16
        assert (k_shot + k_query) <= num_instance

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        self.labels = {"train": self.y_train, "test": self.y_test}
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"],self.labels["train"]),  
                               "test": self.load_data_cache(self.datasets["test"],self.labels["test"])}

    def load_data_cache(self, data_pack, label_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, num_instance]
        :param label_pack:
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes
            
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(self.num_instance, self.k_shot + self.k_query, False)
                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append(label_pack[cur_class][selected_img[:self.k_shot]])
                    y_qry.append(label_pack[cur_class][selected_img[self.k_shot:]])
                    # y_spt.append([j for _ in range(self.k_shot)])
                    # y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.window*self.bit_size)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot, self.bit_size)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.window*self.bit_size)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query, self.bit_size)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.window*self.bit_size)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz, self.bit_size)
            # [b, qrysz]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.window*self.bit_size)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz, self.bit_size)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode], self.labels[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch