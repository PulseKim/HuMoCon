
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


class DataParser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.state = []
        self.successor = []
        self.read_data()
        self.convert_to_tensor()
        self.convert_to_numpy()

    def convert_to_tensor(self, ):
        self.tensor_input = tf.convert_to_tensor(self.state, dtype = tf.float32)
        self.tensor_output = tf.convert_to_tensor(self.successor, dtype = tf.int32)

    def convert_to_numpy(self, ):
        self.np_input = np.array(self.state)
        self.np_output = np.array(self.successor)

    def read_data(self, ):
        start_flag = True
        file_in = open(self.file_path, 'r')
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            state = []
            successor = []
            for i in range(0, len_list-1):
                state.append(float(list_y[i]))
            self.state.append(state)
            successor = [int(list_y[len_list-1])]
            # if int(list_y[len_list-1]) == 1:
            #     successor.append(0)
            #     successor.append(1)
            # else:
            #     successor.append(1)
            #     successor.append(0)
            self.successor.append(successor)

class MLP_model(object):
    def __init__(self, dimX, dimY):
        n_hidden_1 = 512
        n_hidden_2 = 256
        n_hidden_3 = 128
        n_hidden_4 = 64
        sigma_init = 0.1
        self.dimX = dimX
        self.dimY = dimY
        self.weights = {
            'h1': tf.Variable(tf.random_normal([dimX, n_hidden_1], stddev = sigma_init)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev = sigma_init)),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev = sigma_init)),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev = sigma_init)),
            'out': tf.Variable(tf.random_normal([n_hidden_4, dimY], stddev = sigma_init))
        }
        sigma_init2 = 0.1
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev = sigma_init2)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev = sigma_init2)),
            'b3': tf.Variable(tf.random_normal([n_hidden_3], stddev = sigma_init2)),
            'b4': tf.Variable(tf.random_normal([n_hidden_4], stddev = sigma_init2)),
            'out': tf.Variable(tf.random_normal([dimY], stddev = sigma_init2))
        }
    def multilayer_perceptron(self, x):
        Layer1 = tf.nn.leaky_relu(tf.add(tf.matmul(x,self.weights['h1']), self.biases['b1']))
        Layer2 = tf.nn.leaky_relu(tf.add(tf.matmul(Layer1,self.weights['h2']), self.biases['b2']))
        Layer3 = tf.nn.leaky_relu(tf.add(tf.matmul(Layer2,self.weights['h3']), self.biases['b3']))
        Layer4 = tf.nn.leaky_relu(tf.add(tf.matmul(Layer3,self.weights['h4']), self.biases['b4']))
        out_layer = tf.nn.sigmoid(tf.add(tf.matmul(Layer4,self.weights['out']), self.biases['out']))
        return out_layer


class SafeOrNot(object):
    def __init__(self, train_data, test_data, directory, save):
        self.train_data = train_data
        self.test_data = test_data
        self.state_train = self.train_data.np_input
        self.output_train = self.train_data.np_output
        self.state_test = self.test_data.np_input
        self.output_test = self.test_data.np_output
        self.dimX = self.state_train.shape[1]
        self.dimY = self.output_train.shape[1]
        self.nTrain = self.state_train.shape[0]
        self.nTest = self.state_test.shape[0]
        self.nb_classes = 2
        # self.model = MLP_model(self.dimX, self.nb_classes)
        self.model = MLP_model(self.dimX, self.dimY)
        # self.normalize_input()
        self.save_file = directory +'/safe/' + save + '/model.ckpt'
        # self.normalize_theta()

    def normalize_input(self,):
        adder = np.ones_like(self.input_train)
        adder = self.train_data.max_len * adder
        self.input_train = self.input_train + adder
        self.input_train = (1. / (2.* self.train_data.max_len)) * self.input_train

        adder2 = np.ones_like(self.input_test)
        adder2 = self.test_data.max_len * adder2
        self.input_test = self.input_test + adder2
        self.input_test = (1. / (2.* self.test_data.max_len)) * self.input_test
        # self.input_train = tf.math.scalar_mul()

    def level_confidence(self, Y, Label):
        pass

    def log_barrier(self, Y, t):
        coeff = -1 / t
        phi = tf.math.scalar_mul(coeff, tf.math.log(-Y + tf.math.scalar_mul(1.00001, tf.ones_like(Y))))
        return phi

    def log_barrier_sq(self, Y, t):
        coeff = -1 / t
        phi = tf.math.scalar_mul(coeff, tf.math.log(-Y + tf.math.scalar_mul(1.001, tf.ones_like(Y))))
        return tf.math.square(phi)

    def conditional(self, label, prediction):
        condition = tf.math.equal(label, 1.)
        mse = tf.math.scalar_mul(0.5, tf.square(label - prediction))
        lb = self.log_barrier(prediction, 0.5)
        error = tf.where(condition, mse, lb)
        return error

    def load_ensemble(self, save_dir, file1, file2, file3):
        X = tf.placeholder(tf.float32, shape=[None, self.dimX], name="input")
        Y = tf.placeholder(tf.float32, shape=[None, self.dimY], name="output")
        Y_result = tf.placeholder(tf.float32, shape=[None, self.dimY], name="result")
        Y_pred = self.model.multilayer_perceptron(X)
        loss = tf.reduce_mean(self.conditional(Y, Y_pred))
        loss_result = tf.reduce_mean(self.conditional(Y, Y_result))

        correct_prediction = tf.equal(tf.math.round(Y_pred), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        correct_prediction_result = tf.equal(Y_result, Y)
        accuracy_result = tf.reduce_mean(tf.cast(correct_prediction_result, "float"))

        level_of_confidence = 0.5
        Y_correction = tf.ones_like(Y_pred)
        Y_correction = tf.math.scalar_mul(level_of_confidence, Y_correction)
        Y_correction = tf.subtract(Y_pred, Y_correction)
        safe_predictor = tf.maximum(tf.math.ceil(Y_correction)- Y, tf.zeros_like(Y))
        safe_accuracy = 1.0 - tf.reduce_mean(tf.cast(safe_predictor, "float"))

        Y_correction_result = tf.ones_like(Y_result)
        Y_correction_result = tf.math.scalar_mul(level_of_confidence, Y_correction_result)
        Y_correction_result = tf.subtract(Y_result, Y_correction_result)
        safe_predictor_result = tf.maximum(tf.math.ceil(Y_correction_result)- Y, tf.zeros_like(Y))
        safe_accuracy_result = 1.0 - tf.reduce_mean(tf.cast(safe_predictor_result, "float"))

        saver1 = tf.train.Saver()
        saver2 = tf.train.Saver()
        saver3 = tf.train.Saver()

        sess_1 = tf.Session()
        # sess_1.run(tf.initialize_all_variables())
        saver1.restore(sess_1, save_dir + file1+ '/model.ckpt')
        y1 = sess_1.run(Y_pred, feed_dict={X: self.state_test})
        loss_temp1 = sess_1.run(loss, feed_dict={X: self.state_test, Y:self.output_test}) 
        accuracy_temp1 = sess_1.run(accuracy, feed_dict={X: self.state_test, Y:self.output_test})
        safe_accuracy_temp1 = sess_1.run(safe_accuracy, feed_dict={X: self.state_test, Y:self.output_test})
        print("[Loss / Tranining Accuracy] {:05.4f} / {:05.4f}".format(loss_temp1, accuracy_temp1))
        print("[Safe Accuracy] {:05.4f}".format(safe_accuracy_temp1))
        sess_1.close()

        sess_2 = tf.Session()
        # sess_2.run(tf.initialize_all_variables())
        saver2.restore(sess_2, save_dir + file2+ '/model.ckpt')
        y2 = sess_2.run(Y_pred, feed_dict={X: self.state_test})
        loss_temp2 = sess_2.run(loss, feed_dict={X: self.state_test, Y:self.output_test}) 
        accuracy_temp2 = sess_2.run(accuracy, feed_dict={X: self.state_test, Y:self.output_test})
        safe_accuracy_temp2 = sess_2.run(safe_accuracy, feed_dict={X: self.state_test, Y:self.output_test})
        print("[Loss / Test Accuracy] {:05.4f} / {:05.4f}".format(loss_temp2, accuracy_temp2))
        print("[Safe Accuracy] {:05.4f}".format(safe_accuracy_temp2))
        sess_2.close()

        sess_3 = tf.Session()
        # sess_3.run(tf.initialize_all_variables())
        saver3.restore(sess_3, save_dir + file3+ '/model.ckpt')
        y_original = sess_3.run(Y, feed_dict={Y:self.output_test})
        y3 = sess_3.run(Y_pred, feed_dict={X: self.state_test})
        loss_temp3 = sess_3.run(loss, feed_dict={X: self.state_test, Y:self.output_test}) 
        accuracy_temp3 = sess_3.run(accuracy, feed_dict={X: self.state_test, Y:self.output_test})
        safe_accuracy_temp3 = sess_3.run(safe_accuracy, feed_dict={X: self.state_test, Y:self.output_test})
        print("[Loss / Test Accuracy] {:05.4f} / {:05.4f}".format(loss_temp3, accuracy_temp3))
        print("[Safe Accuracy] {:05.4f}".format(safe_accuracy_temp3))
        
        # y12 = np.multiply(y1,y2)
        # y_result = np.multiply(y12, y3)

        y12 = np.multiply(np.around(y1), np.around(y2))
        y_result = np.multiply(y12, np.around(y3))
        # Y_r = tf.convert_to_tensor(y_result)
        loss_r = sess_3.run(loss_result, feed_dict={Y:self.output_test, Y_result: y_result}) 
        accuracy_r = sess_3.run(accuracy_result, feed_dict={Y:self.output_test, Y_result: y_result})
        safe_accuracy_r = sess_3.run(safe_accuracy_result, feed_dict={Y:self.output_test, Y_result: y_result})

        print("[Loss / Test Accuracy] {:05.4f} / {:05.4f}".format(loss_r, accuracy_r))
        print("[Safe Accuracy] {:05.4f}".format(safe_accuracy_r))
        sess_3.close()
        


    def train(self, ):
        X = tf.placeholder(tf.float32, shape=[None, self.dimX], name="input")
        Y = tf.placeholder(tf.float32, shape=[None, self.dimY], name="output")
        Y_pred = self.model.multilayer_perceptron(X)
        loss = tf.reduce_mean(self.conditional(Y, Y_pred))
        # loss = tf.reduce_mean(tf.math.scalar_mul(0.5, tf.square(Y - Y_pred)))
        # loss = tf.compat.v1.losses.huber_loss(Y_pred, Y, delta = 0.5)

        learning_rate = 5e-5
        adam = tf.train.AdamOptimizer(learning_rate= learning_rate)
        optimizer = adam.minimize(loss)
        training_epochs = 501
        display_epoch = 10
        batch_size = 512

        correct_prediction = tf.equal(tf.math.round(Y_pred), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # correct_prediction = tf.equal((Y_pred, 1), tf.argmax(Y, 1))
        level_of_confidence = 0.5
        Y_correction = tf.ones_like(Y_pred)
        Y_correction = tf.math.scalar_mul(level_of_confidence, Y_correction)
        Y_correction = tf.subtract(Y_pred, Y_correction)
        correct_prediction_mod = tf.equal(tf.math.ceil(Y_correction), Y)        
        safe_predictor = tf.maximum(tf.math.ceil(Y_correction)- Y, tf.zeros_like(Y))
        safe_accuracy = 1.0 - tf.reduce_mean(tf.cast(safe_predictor, "float"))
        saver = tf.train.Saver()
        prev_loss = 1E5
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(training_epochs):
                nBatch  = int(self.nTrain/batch_size)
                myIdx =  np.random.permutation(self.nTrain)
                for ii in range(nBatch):
                    X_batch = self.state_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
                    Y_batch = self.output_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
                    sess.run(optimizer, feed_dict={X:X_batch, Y:Y_batch})
                # print("Y batch")
                # print(Y_batch)
                # print("One hot")
                # print(sess.run(Y_one_hot, feed_dict={Y:Y_batch}))
                # print("Pred")
                # print(sess.run(Y_pred, feed_dict={X:X_batch}))

                if (epoch) % display_epoch == 0:
                    loss_temp = sess.run(loss, feed_dict={X: self.state_train, Y:self.output_train}) 
                    accuracy_temp = accuracy.eval({X: self.state_train, Y:self.output_train})
                    safe_accuracy_temp = safe_accuracy.eval({X: self.state_train, Y:self.output_train})
                    print("(epoch {})".format(epoch))
                    print("[Loss / Tranining Accuracy] {:05.4f} / {:05.4f}".format(loss_temp, accuracy_temp))
                    print("[Safe Accuracy] {:05.4f}".format(safe_accuracy_temp))
                    print(" ")
                    if abs(prev_loss - loss_temp) < 5e-5:
                        break
                    prev_loss = loss_temp
            saver.save(sess, self.save_file)
            loss_test = sess.run(loss, feed_dict={X: self.state_test, Y:self.output_test})     
            accuracy_test = accuracy.eval({X: self.state_test, Y:self.output_test})
            safe_accuracy_test = safe_accuracy.eval({X: self.state_test, Y:self.output_test})
            print("[Test Loss / Test Accuracy] {:05.4f} / {:05.4f}".format(loss_test, accuracy_test))    
            print("[Safe Accuracy] {:05.4f}".format(safe_accuracy_test))
            sess.close()
            # temp = np.array([[0.427, 0.014, -0.139]])
            # np.reshape(temp, (-1,3))
            # adder = np.ones_like(temp)
            # adder = self.train_data.max_len * adder
            # temp = temp + adder
            # temp= (1. / (2.* self.train_data.max_len)) * temp
            # print(temp)
            # print([-0.533, -0.823, -0.981])
            # temp2 = sess.run(Y_pred, feed_dict={X:temp})
            # print(temp2)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    rscdir = current_dir+ "/env/rsc"
    # train_data1 = DataParser(rscdir + "/ensemble1.txt")
    # train_data2 = DataParser(rscdir + "/ensemble2.txt")
    train_data3 = DataParser(rscdir + "/ensemble3.txt")  
    test_data = DataParser(rscdir + "/ensemble_all.txt")
    safety_net1 = SafeOrNot(train_data3, test_data, current_dir, "ensemble3")

    # safety_net1.train()
    safety_net1.load_ensemble(current_dir+'/safe/', 'ensemble1', 'ensemble2', 'ensemble3')
