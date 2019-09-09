'''
Created on 14 Aug 2019

Code to invert an input
feature at FC8 layer of SVDNet. 
For more information refer to EUSIPCO 2018 paper.
@https://ieeexplore.ieee.org/abstract/document/8553178
@author: Saumitra
'''

import io
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
floatX = theano.config.floatX
import utils
from progress import progress
from simplecache import cached
import audio
import model
import upconv
import plots
import csv

def main():
    # parse the command line arguments
    parser = utils.argument_parser()
    args = parser.parse_args()
    
    print("-------------------------------")
    print("classifier:%s" %args.classifier)
    print("inverter:%s" %args.inverter)
    print("dataset_path:%s" %args.dataset_path)
    print("dataset name:%s" %args.dataset)
    print("results path:%s" %args.results_dir)
    print("quantitative analysis:%s" %args.quant_analysis)
    print("mask inversion flag: %r" % args.mask_inv_flag)
    print("plot quant results case: %r" %args.plot_quant_res)
    print("-------------------------------")
    
    # just plots the quantitative analysis results and exits
    if args.plot_quant_res:
        # jamendo results
        exp_loss_jamendo_case1 = [0, 6.48, 10.87, 13.33, 15.51, 19.15, 25.94, 37.56, 49.11, 56.85, 57.77]#, 57.77]
        exp_loss_jamendo_case2 = [57.77, 59.19, 58.11, 51.81, 43.1, 31.87, 22.84, 15.51, 11.03, 5.86, 0.03]#, 0]
        rel_area_jamendo_case1 = [100, 96, 87, 77, 65, 53, 39, 26, 14, 4, 0]#, 0]
        rel_area_jamendo_case2 = [0, 4, 13, 23, 35, 47, 61, 74, 86, 96, 100]#, 100]
        exp_losses_jamendo = [exp_loss_jamendo_case1, exp_loss_jamendo_case2]
        rel_areas_jamendo = [rel_area_jamendo_case1, rel_area_jamendo_case2]
        
        # rwc results
        exp_loss_rwc_case1 = [0, 6.52, 10.9, 13.39, 15.87, 21.28, 30.92, 43.22, 53.41, 60.85, 63.66]#, 63.66]
        exp_loss_rwc_case2 = [63.66, 64.5, 61.01, 52.55, 39.39, 26.27, 16.13, 9.55, 5.05, 2.26, 0.03]#, 0]
        rel_area_rwc_case1 = [100, 96, 87, 75, 61, 47, 33, 20, 10, 3, 0]#, 0]
        rel_area_rwc_case2 = [0, 4, 13, 25, 39, 53, 67, 80, 90, 97, 100]#, 100]
        exp_losses_rwc = [exp_loss_rwc_case1, exp_loss_rwc_case2]
        rel_areas_rwc = [rel_area_rwc_case1, rel_area_rwc_case2]
        
        plots.quant_eval(exp_losses_jamendo, rel_areas_jamendo, exp_losses_rwc, rel_areas_rwc, args.results_dir)
        exit(0)
    
    # default parameters 
    sample_rate = 22050
    frame_len = 1024
    fps = 70
    mel_bands = 80
    mel_min = 27.5
    mel_max = 8000
    blocklen = 115
    batchsize = 32
    
    # single instance inversion/quantitative analysis parameters
    preds_before = []
    if not args.quant_analysis:
        time_index = 10
        masking_threshold = [0.7]
        duration = 0 # no use
        increment = 0.5
    else:
        preds_after = []
        area_per_instance = []
        result = []
        start_offset = 5
        end_offset = 20
        duration = 200
        increment = 0.5
        masking_threshold = np.arange(0.0, 1.2, 0.1)
        class_threshold = 0.66 # Calculated over Jamendo validation dataset

    # printing and plotting parameters
    df=True
    #inp = []
    #expns =[]
  
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
    
    # prepare dataset
    datadir = os.path.join(os.path.dirname(__file__), args.dataset_path, 'datasets', args.dataset)
    
    # load filelist
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]

    # compute spectra
    print("Computing%s spectra..." %
          (" or loading" if args.cache_spectra else ""))
    
    spects = [] # list of tuples, where each tuple has magnitude and phase information for one audio file
    for fn in progress(filelist, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))
        
    # prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)
    
    # precompute mel spectra, if needed, otherwise just define a generator
    mel_spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank), 1e-7)) for spect in spects)
    
    # load mean/std or compute it, if not computed yet
    meanstd_file = os.path.join(os.path.dirname(__file__), '%s_meanstd.npz' % args.dataset)
    with np.load(meanstd_file) as f:
            mean = f['mean']
            std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)
    
    print("Preparing training data feed...")
    # normalised mel spects, without data augmentation
    mel_spects = [(spect - mean) * istd for spect in mel_spects]
    
    # we create two theano functions
    # the first one uses pre-trained classifier to generate features and predictions
    # the second one uses pre-trained inverter to generate mel spectrograms from input features 
    
    # classifier (discriminator) model
    input_var = T.tensor3('input')
    inputs = input_var.dimshuffle(0, 'x', 1, 2)  # insert "channels" dimension, changes a 32 x 115 x 80 input to 32 x 1 x 115 x 80 input which is fed to the CNN
    
    network = model.architecture(inputs, (None, 1, blocklen, mel_bands))
    
    # load saved weights
    with np.load(args.classifier) as f:
        lasagne.layers.set_all_param_values(
                network['fc9'], [f['param%d' % i] for i in range(len(f.files))])
        
    # create output expression
    outputs_score = lasagne.layers.get_output(network['fc8'], deterministic=True)
    outputs_pred = lasagne.layers.get_output(network['fc9'], deterministic=True)

    # prepare and compile prediction function
    print("Compiling classifier function...")
    pred_fn_score = theano.function([input_var], outputs_score, allow_input_downcast= True)
    pred_fn = theano.function([input_var], outputs_pred, allow_input_downcast= True)
    
    # inverter (generator) model    
    input_var_deconv = T.matrix('input_var_deconv')

    # inverter (generator) model    
    gen_network = upconv.architecture_upconv_fc8(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network['fc8'])[1]))
    
    # load saved weights
    with np.load(args.inverter) as f:
        lasagne.layers.set_all_param_values(
                gen_network, [f['param%d' % i] for i in range(len(f.files))])
    
    # create cost expression
    outputs = lasagne.layers.get_output(gen_network, deterministic=True)
    print("Compiling inverter function...")
    test_fn = theano.function([input_var_deconv], outputs, allow_input_downcast= True)
    
    # instance-based feature inversion
    # (1) pick a file from a dataset (e.g., dataset: Jamendo test) (2) select a time index to read the instance
    file_idx = np.arange(0, len(filelist))
    hop_size= sample_rate/fps # samples
    
    for mt in masking_threshold:
        
        np.random.seed(0)

        print("\n ++++++ Masking threshold: %f +++++\n " %(mt))
        
        for file_instance in file_idx:        
            
            print("<<<<Analysis for the file: %d>>>>" %(file_instance+1))
            
            if args.quant_analysis:
                time_idx = np.random.randint(start_offset, end_offset, 1)[0]   # provides a random integer start position between start and end offsets            
            else:
                time_idx = time_index
            
            td = time_idx # no use for the single instance inversion case.
                
            # generate excerpts for the selected file_idx
            # excerpts is a 3-d array of shape: num_excerpts x blocklen x mel_spects_dimensions   
            num_excerpts = len(mel_spects[file_instance]) - blocklen + 1
            print("Number of excerpts in the file :%d" %num_excerpts)
            excerpts = np.lib.stride_tricks.as_strided(mel_spects[file_instance], shape=(num_excerpts, blocklen, mel_spects[file_instance].shape[1]), strides=(mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[1]))
            
            while(time_idx<= td+duration):
                # convert the time_idx to the excerpt index
                excerpt_idx = int(np.round((time_idx * sample_rate)/(hop_size)))
                print("Time_idx: %.2f secs, Excerpt_idx: %d" %(time_idx, excerpt_idx))
                if ((excerpt_idx + batchsize) > num_excerpts):
                    print("------------------Number of excerpts are less for file: %d--------------------" %(file_instance+1))
                    break
            
                # generating feature representations for the select excerpt.
                # CAUTION: Need to feed mini-batch to the pre-trained model, so (mini_batch-1) following excerpts are also fed, but are not analysed
                # classifier can have less than minibatch data, but the inverter needs a batch of data to make prediction (comes from how the inverter was trained)
                scores = pred_fn_score(excerpts[excerpt_idx:excerpt_idx + batchsize])
                #print("Feature"),
                #print(scores[file_idx])
                
                predictions = pred_fn(excerpts[excerpt_idx:excerpt_idx + batchsize])
                print("Prediction score for the excerpt without masking:%f" %(predictions[0][0]))
                preds_before.append(predictions[0][0])
                
                mel_predictions = np.squeeze(test_fn(scores), axis = 1) # mel_predictions is a 3-d array of shape batch_size x blocklen x n_mels
                
                # normalising the inverted mel to create a map, and use the map to cut the section in the input mel
                norm_inv = utils.normalise(mel_predictions[0])
                norm_inv[norm_inv<mt] = 0 # Binary mask----- 
                norm_inv[norm_inv>=mt] = 1

                if args.quant_analysis:
                    # calculate area
                    area = utils.area_calculation(norm_inv, debug_flag = df)
                    
                    # reversing the mask to keep the portions that seem not useful for the current instance prediction
                    norm_inv, area = utils.invert_mask(mask = norm_inv, mask_inv_flag = args.mask_inv_flag, area_mask = area, debug_flag=df)

                # masking out the input based on the mask created above
                masked_input = np.zeros((batchsize, blocklen, mel_bands))
                masked_input[0] = norm_inv * excerpts[excerpt_idx]
                
                if args.quant_analysis:                    
                    # save the area enabled
                    area_per_instance.append(area)
                    # feed the updated input to regenerate prediction
                    # just changing the first input.
                    predictions = pred_fn(masked_input)
                    print("Predictions score for the excerpt after masking:%f\n" %(predictions[0][0]))
                    preds_after.append(predictions[0][0])
                
                if not args.quant_analysis: # save results
                    # saves plots for the input Mel spectrogram and its inverted representation
                    # all plots are normalised in [0, 1] range
                    plots.single_inst_inv(utils.normalise(excerpts[excerpt_idx]), utils.normalise(mel_predictions[0]), norm_inv, utils.normalise(masked_input[0]), preds_before[0], file_instance, excerpt_idx, args.results_dir, 'FC8')

                time_idx += increment
            
            # plotting figure 6.4 in thesis              
            #plots.input_mels()
            
            # plotting figure 6.6 in thesis
            '''inp.append(excerpts[excerpt_idx])
            expns.append(masked_input[0])
            preds_before.append(predictions[0][0])
        plots.special_cases(utils.normalise(inp[0]), utils.normalise(expns[0]), utils.normalise(inp[1]), utils.normalise(expns[1]), preds_before[0], preds_before[1], file_instance, excerpt_idx, args.results_dir)'''
        
        if args.quant_analysis:
            res_tuple = utils.quant_result_analysis(preds_before, preds_after, area_per_instance, mt, class_threshold, debug_flag=df)
            result.append(res_tuple) # one result per threshold value
            
        # clearing the lists for the next iteration
        preds_before = []
        preds_after = []
        area_per_instance = []
    
    if args.quant_analysis:
        # save the quantitative analysis results
        quant_result_columns = ['threshold', 'total instances', 'total fails', 'explanation loss [%]', 'average area']
        with open(args.results_dir + '/' + 'quant_analysis_result.csv', 'w') as fp:
            results_writer = csv.writer(fp, delimiter=',')
            results_writer.writerow(quant_result_columns)
            for result_th in result:
                results_writer.writerow(result_th)
            

if __name__ == '__main__':
    main()


