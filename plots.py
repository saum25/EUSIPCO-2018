'''
Created on 9 Dec 2017

@author: Saumitra
'''

import matplotlib.pyplot as plt
import librosa.display as disp
import numpy as np

def single_inst_inv(input_excerpt, inv, mask, masked_input, pred, file_id, excerpt_id, results_path, layer):
    """
    plots the figure 3 from the EUSIPCO paper
    """

    fs = 6
    plt.figure(figsize=(4,4))
    
    plt.subplot(2, 2, 1)
    disp.specshow(input_excerpt.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'coolwarm')
    plt.title('(A)', fontsize = fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.yticks(fontsize = fs)
    plt.xticks(fontsize = fs)
    
    plt.subplot(2, 2, 2)
    disp.specshow(inv.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
    plt.title('(B)', fontsize = fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
       
    plt.subplot(2, 2, 3)
    disp.specshow(mask.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
    plt.title('(C)', fontsize=fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)

    plt.subplot(2, 2, 4)
    disp.specshow(masked_input.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
    plt.title('(D)', fontsize = fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    
    plt.subplots_adjust(bottom=0.125, wspace=0.18, hspace=0.18)
    cax = plt.axes([0.93, 0.125, 0.02, 0.775])
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=fs)

    
    #plt.tight_layout()
    plt.savefig(results_path + '/'+ 'plot'+ '_fileid_'+ str(file_id) + '_excerptid_' + str(excerpt_id) + '_pred_'+ "%.2f"  %pred +'.pdf', dpi = 300)
    
def input_mels():
    """
    plots figure 6.4 in the thesis
    """
    path1 = 'results/nonvocal_1.4825sec.npz'
    path2 = 'results/vocal_33sec.npz'
    fs=9
    
    with np.load(path1) as fp:
        # list of np arrays
        ana_data1 = [fp[ele] for ele in sorted(fp.files)]

    with np.load(path2) as fp:
        # list of np arrays
        ana_data2 = [fp[ele] for ele in sorted(fp.files)]
        
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    disp.specshow(ana_data1[0].T, x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('Freq(Hz)', labelpad=0.5, fontsize=fs)
    plt.xlabel('Time(sec)', labelpad=0.5, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('Non-vocal excerpt', fontsize=fs)

    plt.subplot(1, 2, 2)
    disp.specshow(ana_data2[0].T, x_axis='time', hop_length= 315, y_axis= 'mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)', labelpad=1, fontsize=fs)
    plt.ylabel('Freq(Hz)', labelpad=1, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('Vocal excerpt', fontsize=fs)    
    
    plt.subplots_adjust(bottom=0.125, wspace=0.08, hspace=0.08)   
    cax = plt.axes([0.92, 0.125, 0.0150, 0.775])
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=fs)
    plt.savefig('results/inp.pdf', dpi=300)
    
def special_cases(inp0, exp0, inp1, exp1, p0, p1, file_id, excerpt_id, results_path):
    """
    plots figure 6.6 in the thesis
    """
    
    fs = 6
    plt.figure(figsize=(4,4))
    
    plt.subplot(2, 2, 1)
    disp.specshow(inp0.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'coolwarm')
    plt.title('(A)', fontsize = fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.yticks(fontsize = fs)
    plt.xticks(fontsize = fs)
    
    plt.subplot(2, 2, 2)
    disp.specshow(exp0.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
    plt.title('(B)', fontsize = fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
       
    plt.subplot(2, 2, 3)
    disp.specshow(inp1.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
    plt.title('(C)', fontsize=fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)

    plt.subplot(2, 2, 4)
    disp.specshow(exp1.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
    plt.title('(D)', fontsize = fs)
    plt.xlabel('Time(sec)', fontsize = fs, labelpad = 1)
    plt.ylabel('Freq(Hz)', fontsize = fs, labelpad = 1)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    
    plt.subplots_adjust(bottom=0.125, wspace=0.18, hspace=0.18)
    cax = plt.axes([0.93, 0.125, 0.02, 0.775])
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=fs)

    plt.savefig(results_path + '/'+ 'plot'+ '_fileid_'+ str(file_id) + '_excerptid_' + str(excerpt_id) + '_pred0_'+ "%.2f"  %p0 + '_pred1_'+ "%.2f"  %p1 +'.pdf', dpi = 300)
        
    
def quant_eval(exp_loss_jamendo, rel_area_jamendo, exp_loss_rwc, rel_area_rwc, result_dir):
    """
    plots the quantitative evaluation results for the jamendo and rwc datasets 
    each argument is a list of two lists - the first one contains results for 
    the case when masked input is fed back to SVDNet and the second one contains
    results for the case when the mask is inverted. The functions plots two
    figures one for each dataset results.
    """
    
    thresholds = np.arange(0.0, 1.1, 0.1)    
    plt.figure(1)
    a = 0.60
    lw = 2
    m = 8
    fs = 10
    lp = 2
    
    # plot explanation loss
    ax1 = plt.subplot(211)
    ax1.plot(thresholds, exp_loss_jamendo[0], color='r', marker='.', mew = 4, alpha=a, linewidth=lw, label='$M^{Jamendo}_1$') # 0th list->normal mask case
    ax1.plot(thresholds, exp_loss_jamendo[1], color='g', marker='|', mew = 3, ms = m, alpha=a, linewidth=lw, label='$M^{Jamendo}_2$') # 1st list->inverted mask case
    ax1.plot(thresholds, exp_loss_rwc[0], color='b', marker='.', mew = 4, alpha=a, linewidth=lw, label='$M^{RWC}_1$') # 0th list->normal mask case    
    ax1.plot(thresholds, exp_loss_rwc[1], color='m', marker='|', mew = 3, ms = m, alpha=a, linewidth=lw, label='$M^{RWC}_2$') # 1st list->inverted mask case
    plt.setp(ax1.get_xticklabels(), visible=False) # turns of the xtick labels https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/shared_axis_demo.html
    #ax1.set_xlabel('masking threshold', labelpad=lp)
    ax1.set_ylabel('explanation loss [%]', labelpad=lp)
    ax1.set_ylim(0, 100)
    plt.title('(A)')
    ax1.grid()
    ax1.legend(fontsize=fs, loc="upper center")

    # plot relative area    
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(thresholds, rel_area_jamendo[0], color='r', marker='.', mew = 4, alpha=a, linewidth=lw, label='$M^{Jamendo}_1$') # 0th list->normal mask case
    ax2.plot(thresholds, rel_area_jamendo[1], color='g', marker='|', mew = 3, ms = m, alpha=a, linewidth=lw, label='$M^{Jamendo}_2$')  # 1st list->inverted mask case
    ax2.plot(thresholds, rel_area_rwc[0], color='b', marker='.', mew = 4, alpha=a, linewidth=lw, label='$M^{RWC}_1$') # 0th list->normal mask case
    ax2.plot(thresholds, rel_area_rwc[1], color='m', marker='|', mew = 3, ms = m, alpha=a, linewidth=lw, label='$M^{RWC}_2$')  # 1st list->inverted mask case
    ax2.set_xlabel('masking threshold', labelpad=lp)
    ax2.set_ylabel('average relative area [%]', labelpad=lp)
    plt.title('(B)')
    ax2.grid()
    ax2.legend(fontsize=fs, loc="center right")
    
    plt.tight_layout()
    plt.savefig(result_dir + '/' + 'feat_inv_quant_res.pdf', dpi=300)
    
  

