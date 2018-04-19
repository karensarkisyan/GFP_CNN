from classes import *
import time
import matplotlib
from optparse import OptionParser

matplotlib.use('Agg')
from itertools import product
from matplotlib import pyplot as plt

num_iter = 3
mse_train = []
mse_val = []

parser = OptionParser()
parser.add_option("-n", "--n_of_scale_block_combination", type="int",
                  help="Scale/block combination choice",
                  dest="n_of_scale_block_combination")

(options, args) = parser.parse_args()

n = options.n_of_scale_block_combination
num_scales, block_repeats = choose_scales_blocks_combination(n)
mode = 'gpu'
n_epoch = 100
kernel_size = 3
pool_size = 3

variable_tested_1 = [0, 0.1, 0.2, 0.3, 0.4]
variable_tested_2 = [0.1*x for x in range(5, 11)]

timestr = time.strftime("%Y%m%d-%H%M")
log_dir = '../models/' + timestr + '/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

f = '../data/amino_acid_genotypes_to_brightness.txt'
batch_size, zero_sample_fraction = 100, 0.5
input_data = Data(file_path=f, batch_size=batch_size, zero_sample_fraction=zero_sample_fraction, zeroing=True)

for it, var in enumerate(product(variable_tested_1, variable_tested_2)):

    print('ITERATION #', it)

    NN_name = 'ResNet'+str(num_scales)+'_'+str(batch_size)
    weight_decay = var[0]
    keep_prob = var[1]

    NN_id = "S%dB%d_WD%.2f_DO%.2f" % (num_scales, block_repeats, weight_decay, keep_prob)

    reset_graph()
    try:
        nn_instance = ResNet(input_data, num_scales, block_repeats, NN_name, mode,
                             kernel_size, pool_size, weight_decay, keep_prob, n_epoch)

        train_mse_hist, val_mse_hist = train_NN(nn_instance, input_data, 20, log_dir, NN_id)

        mse_train.append(train_mse_hist[-1])
        mse_val.append(val_mse_hist[-1])

    except:
        mse_train.append(0)
        mse_val.append(0)

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

results = pd.DataFrame([mse_train, mse_val], columns=list(product(variable_tested_1, variable_tested_2)),
                       index=['Train', 'Test'])
results.to_csv('../tmp/' + timestr + '_' + '_'.join(
    NN_id.split('_')[:-1]) + '_mse.txt', sep='\t')
