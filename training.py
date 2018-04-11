from classes import *
from collections import OrderedDict
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

num_iter=3
mse_train=[]
mse_val=[]

variable_tested = [0.001,0.01,0.1,0.2,0.5]

timestr = time.strftime("%Y%m%d-%H")
log_dir = '../models/' + timestr + '/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

for it,var in enumerate(variable_tested):
    clear_output()
    print 'ITERATION #', it

    #num_scales=int(it/3)+1
    num_scales=1
    block_repeats=3
    NN_name='ResNet'
    mode='gpu'
    kernel_size=3
    pool_size=3
    weight_decay=var
    keep_prob=0.8
    n_epoch=100

    NN_id="Weight_decay_"+str(var)
    
    reset_graph()
    f='../data/amino_acid_genotypes_to_brightness.txt'
    batch_size,zero_sample_fraction = 100, 0.5

    input_data = Data(file_path=f,batch_size=batch_size,zero_sample_fraction=zero_sample_fraction, zeroing=True)
    
    nn_instance = ResNet(input_data, num_scales, block_repeats, NN_name, mode,
                     kernel_size, pool_size, weight_decay, keep_prob, n_epoch)
    
    train_mse_hist, val_mse_hist = train_NN(nn_instance, input_data, 20, log_dir, NN_id)
    
    mse_train.append(train_mse_hist[-1])
    mse_val.append(val_mse_hist[-1])

    ######CREATING THE TEST SET IN THE FIRST ITERATION######

    if it==0:
        print 'Generating data for prediction'

        wt_sq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
        unique_mutations = []
        for mut in input_data.mutant_list:
            unique_mutations.extend(mut)

        unique_mutations = [[x] for x in set(unique_mutations)]
        
        #start the list with a wild-type
        sq_test_set = [wt_sq]

        for mutant in unique_mutations:
            sq_test_set.append(make_mutant_sq(wt_sq, mutant))
            
        #creating the initial 0-filled carcas of unfolded versions of sqs, which will be further turned into binary matricies of shape 238 by 20 (aas)
        unfolded_df = OrderedDict()

        for aa in set([item[-1] for sublist in unique_mutations for item in sublist]):
            unfolded_df[aa] = np.zeros((len(sq_test_set), len(wt_sq)))

        #filling the binary matrices, corresponding to 20 different amino acids within the unfoded_df dict
        for ind,mutant in enumerate(sq_test_set):
            for pos,mut in enumerate(mutant):
                unfolded_df[mut][ind, pos] = 1.  
                
        #stacking all the amino acids into one np array
        input_df = np.stack(unfolded_df.values(),axis=1)

        #putting the channel info (amino acids) to the end
        input_df = np.swapaxes(input_df,-1,-2)

    ############

    print 'Deleting input data'
    del input_data
    
    # print 'Restoring session for prediction'
    # with tf.Session() as sess:
    #
    #     saver = tf.train.Saver()
    #     saver.restore(sess, log_dir+"model.ckpt")
    #     predictions_test = sess.run(nn_instance.preds_val,{nn_instance.x_val_ph:input_df})
    #
    # if it==0:
    #     recording_predictions = np.zeros(shape=[len(input_df), len(variable_tested)])
    #
    # for i,val in enumerate(predictions_test):
    #     if val[0] > 3.72:
    #         recording_predictions[i,it]=1
    #
    # print 'Writing results to file'
    # np.save('../tmp/'+meta_timestr+'_predictions.npy',recording_predictions)

results=pd.DataFrame([mse_train,mse_val],columns=variable_tested,index=['Train','Test'])
results.to_csv('../tmp/'+timestr+'_mse_recorded.txt',sep='\t')