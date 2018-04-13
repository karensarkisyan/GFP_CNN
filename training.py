from classes import *
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

num_iter = 3
mse_train = []
mse_val = []

variable_tested = [0.2,0.22,0.25,0.28,0.3]

timestr = time.strftime("%Y%m%d-%H%M")
log_dir = '../models/' + timestr + '/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

f = '../data/amino_acid_genotypes_to_brightness.txt'
batch_size, zero_sample_fraction = 100, 0.5
input_data = Data(file_path=f, batch_size=batch_size, zero_sample_fraction=zero_sample_fraction, zeroing=True)

for it, var in enumerate(variable_tested):

    print('ITERATION #', it)

    # num_scales=int(it/3)+1
    num_scales = 1
    block_repeats = 3
    NN_name = 'ResNet'
    mode = 'gpu'
    kernel_size = 3
    pool_size = 3
    weight_decay = var
    keep_prob = 0.8
    n_epoch = 100

    NN_id = "Weight_decay_tuning_" + str(var)

    reset_graph()

    nn_instance = ResNet(input_data, num_scales, block_repeats, NN_name, mode,
                         kernel_size, pool_size, weight_decay, keep_prob, n_epoch)

    train_mse_hist, val_mse_hist = train_NN(nn_instance, input_data, 20, log_dir, NN_id)

    mse_train.append(train_mse_hist[-1])
    mse_val.append(val_mse_hist[-1])

    # if it == 0:
    #     print('Generating data for prediction')
    #     input_df = make_data_for_prediction(input_data)
    #     recording_predictions = np.zeros(shape=[len(input_df), len(variable_tested)])
    #
    # print('Deleting input data')
    # del input_data
    #
    # print('Restoring session for prediction')
    # with tf.Session() as sess:
    #
    #     saver = tf.train.Saver()
    #     saver.restore(sess, log_dir + "model_" + NN_id + ".ckpt")
    #     predictions_test = sess.run(nn_instance.preds_val, {nn_instance.x_val_ph: input_df})
    #
    # for i, val in enumerate(predictions_test):
    #     if val[0] > 3.72:
    #         recording_predictions[i, it] = 1
    #
    # print('Writing results to file')
    # np.save('../tmp/' + timestr + '_predictions.npy', recording_predictions)

results = pd.DataFrame([mse_train, mse_val], columns=variable_tested, index=['Train', 'Test'])
results.to_csv('../tmp/' + timestr + '_' + '_'.join(NN_id.split('_')[:-1])+'_mse.txt', sep='\t')
