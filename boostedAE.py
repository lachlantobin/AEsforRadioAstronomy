import sys
import math
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import time

# stuff necessary to read the fits file 

def open_psrfits(in_file):
    hdulist = fits.open(in_file)
    nsblk    = int(hdulist['SUBINT'].header['NSBLK'])
    # Change header if we have an "X" in the bit definition
    colnum = hdulist['SUBINT'].columns.names.index('DATA')
    label_tform = "TFORM"+str(colnum+1)
    label_tdim = "TDIM"+str(colnum+1)
    # Note that I do not know why I need to close and re-open the file after checking
    # this column size.
    hdulist.close()
    hdulist = fits.open(in_file)
    if "X" in hdulist['SUBINT'].header[label_tform]:
        # print(hdulist['SUBINT'].header[label_tform][:-1])
        withoutX = int(int(hdulist['SUBINT'].header[label_tform][:-1])/8)
        hdulist['SUBINT'].header[label_tform]= str(withoutX)+"B"
        nch_orig = int(hdulist['SUBINT'].header[label_tdim].split(",")[0][1:])        
        hdulist['SUBINT'].header[label_tdim]=f"({nch_orig//8},1,{nsblk})"
    return hdulist

def close_psrfits(hdulist):
    hdulist.close()

def read_psrfits_subint(hdulist,isub,apply_scales=True):
    
    # Load in relevant header information
    obsbw    = hdulist['PRIMARY'].header['OBSBW']          
    nbits    = int(hdulist['SUBINT'].header['NBITS'])
    nchan    = int(hdulist['SUBINT'].header['NCHAN'])
    nsblk    = int(hdulist['SUBINT'].header['NSBLK'])
    nsub     = int(hdulist['SUBINT'].header['NAXIS2'])
    npol     = int(hdulist['SUBINT'].header['NPOL'])
    tsamp    = float(hdulist['SUBINT'].header['TBIN'])
    cFreq    = float(hdulist[0].header['OBSFREQ'])   
    stt_imjd = float(hdulist[0].header['STT_IMJD'])  
    stt_smjd = float(hdulist[0].header['STT_SMJD'])  
    stt_offs = float(hdulist[0].header['STT_OFFS'])  

    if npol==2:
        print("Currently cannot process npol = 2 data sets")
        hdulist.close()

        sys.exit()
                 
    # print(f"nchan = {nchan}, nsblk = {nsblk}, tsamp = {tsamp}, nbits = {nbits}, obsbw = {obsbw}, nsub = {nsub}")

    tbdata = hdulist['SUBINT'].data[isub]

    # read in headers 
    hdu_primary_header = hdulist['PRIMARY'].header
    hdu_subint_header = hdulist['SUBINT'].header
    zero_off    = float(hdulist['SUBINT'].header['ZERO_OFF'])
      
    dat_freq = np.reshape(tbdata['DAT_FREQ'], nchan)
    dat_wts = np.reshape(tbdata['DAT_WTS'], nchan)
    dat_offs = np.reshape(tbdata['DAT_OFFS'], (npol, nchan))
    dat_scl = np.reshape(tbdata['DAT_SCL'], (npol, nchan))

    # print(tbdata['DATA'].shape)
    data = (tbdata['DATA'])
    
    if nbits==1:
        data_unpack = np.unpackbits (data)
    elif nbits==2:
        data_unpack = unpack_2bit (data.flatten())
    elif nbits==8:
        data_unpack = data.flatten()
    else:
        print(f"UNABLE TO UNPACK DATA as do not have an unpacker for nbits = {nbits}")
        hdulist.close()
        sys.exit()


    if npol==4:
        print("Forming total intensity from npol=4 data")
        repack = data_unpack.reshape(nsblk,npol,nchan)
        data_final = np.sum(repack[:,:2,:],axis=1).T.astype('float')
    else:    
        data_final = data_unpack.reshape(nsblk,nchan).T.astype('float')
        # print(f"data_final shape = {data_final.shape}, data_scl = {dat_scl.flatten().shape}")
        if apply_scales:
            for ichan in range(0,nchan):
                data_final[ichan,:] = ((data_final[ichan,:]-zero_off)*dat_scl[0][ichan] + dat_offs[0][ichan])   # *(dat_scl[0][ichan]))+dat_offs[0][ichan]
    return data_final

import glob
import keras
import cv2

from scipy.stats import norm
from scipy.stats import zscore

start_time = time.time()

# boosting based training phase

def reconstruction_loss(data, model):
    prediction = {}
    error = {}
    # splitting the data below is just a precautionary measure for large datasets - encountered the problem of running out of memory when trying to reconstruct a lot of data at once
    split_data = np.array_split(data, 6) # 6 is arbitrary
    for i in range(6):
        prediction[i] = model.predict(split_data[i])
        error[i] = np.mean((split_data[i]-prediction[i])**2, axis=1) # calculate reconstruction error
    print("Predicted!")
    reconstruction_error = np.concatenate((error[0], error[1], error[2], error[3], error[4], error[5])) 
    return reconstruction_error

def boost_train(data, model, m, n, dim):
    # data = data in the form of numpy array, m = number of times to retrain the autoencoder, n = how many images we have
    # randomly select original training/testing data for first iteration
    indices = np.array_split(np.random.choice(data.shape[0], n//4, replace=False), 2) # careful if n is too large! 
    train = data[indices[0]]
    test = data[indices[1]]
    # reshape
    train = train.reshape((-1,dim))
    test = test.reshape((-1,dim))
    # we would like to store the reconstruction errors at each stage (to later calculate our weight)
    error_matrix = np.zeros(shape=(m, n))
    for i in range(m):
        print(f"Iteration {i}")
        # first train autoencoder on the ith iteration
        history = model.fit(train, train, epochs=40, batch_size=100, validation_data=(test, test))
        print("Succesfully trained!")
        # now calculation reconstruction errors for each data point x
        losses = reconstruction_loss(data.reshape((-1,dim)), model)
        print("Losses calculated!")
        # print(losses)
        probability = (1 / losses)/np.sum(1/ losses)
        smallest_losses = np.random.choice(list(range(len(losses))), n//8, p=probability)
        # sample n/4 of the dataset using the probability function
        smallest_losses = np.array_split(smallest_losses, 2)
        train = data[smallest_losses[0]]
        test = data[smallest_losses[1]]
        # reshape
        train = train.reshape((-1,dim))
        test = test.reshape((-1,dim))
        error_matrix[i] = np.array(losses)
    return error_matrix

# consensus phase

def consensus(m, n, error_matrix):
    weight = []
    # following "consensus phase" in the Sarvari et al. (2019) paper, where we weight the reconstruction error at each boosting stage to get the final reconstruction error
    for i in range(m):
        numerator = 1/np.sum(error_matrix[i])
        denominator = []
        for i in range(m):
            denominator.append(1/np.sum(error_matrix[i]))
        weight.append(numerator/np.sum(np.array(denominator)))
        print(f"numerator = {numerator}")
        print(f"denominator = {np.sum(np.array(denominator))}")
        print(f"weight = {numerator/np.sum(np.array(denominator))}")
    errors = np.zeros(n)
    print(weight)
    weight = np.array(weight)
    # now we are indexing over each element of the data array
    for i in range(n):
        # weight the error from each iteration i by the "outlieryiness" of each iteration!
        errors[i] = np.sum(weight * error_matrix[:,i])
    return errors

# define model & load size information
files = [f for f in glob.glob(f'{sys.argv[1]}*.sf')]
print(files)
hdulist = open_psrfits(files[0])
reduced = 2 # factor by which to reduce the dimension using cv2.reize (speeds things up significantly) 
testshape = list(read_psrfits_subint(hdulist,0,apply_scales=False).shape)
keras_shape = testshape[0]//reduced*testshape[1]//reduced 

# should play around with number of layers, neurons etc. 
model = keras.Sequential([
    # encoder
    keras.layers.Dense(128, activation='sigmoid', input_shape=(keras_shape,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(8, activation='relu'),

    # decoder
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(keras_shape, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

combined_zscores = {}

j = 1
for f in files:
    
    # load file
    in_file = f
    hdulist = open_psrfits(in_file)
    
    # load data!
    nsub    = int(hdulist['SUBINT'].header['NAXIS2'])
    data_shape = list(read_psrfits_subint(hdulist,0,apply_scales=False).shape)
    data_shape.insert(0, nsub)
    read_data = np.zeros(shape=(nsub,data_shape[1],data_shape[2]))
    data = np.zeros(shape=(nsub,data_shape[1]//reduced,data_shape[2]//reduced))
    dim = data.shape[1]*data.shape[2]
    keras_data = data.reshape((-1,dim))
    print(keras_data.shape[1])
    for i in range(nsub):
        read_data[i] = read_psrfits_subint(hdulist,i,apply_scales=False)
        data[i] = cv2.resize(read_data[i], dsize=(data_shape[2]//reduced, data_shape[1]//reduced), interpolation=cv2.INTER_CUBIC) # reduce dimension by half.

    # done!
    print(f"Data loaded with nsub = {nsub}!")
    print(data_shape)
    
    # call functions
    iterations = 5 # boosting iterations
    error_matrix = boost_train(data, model, iterations, nsub, dim)
    errors = consensus(iterations, nsub, error_matrix)
    print(error_matrix)
    print(errors)

    # create plots + fit normal distribution
    (mu, sigma) = norm.fit(errors)
    n, bins, patches = plt.hist(errors, bins=100, density=True, facecolor='green', alpha=0.75)
    plt.title("Distribution of Reconstruction Errors")
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    # plt.savefig('/DATA/OCTANS_1/tob023/working/simulations/Benchmarking/histogram.jpeg', format='jpeg',  dpi=200)
    plt.savefig(f'/u/tob023/working/P505Again/{iterations}Iterations_File{j}Histogram.jpeg', dpi=250) # change this path if you want to put the histogram somewhere specific
    plt.clf()

    # statistics
    zscores = zscore(errors)
    combined_zscores[j] = zscores
    outliers = np.where(zscores > 4)[0] # vary this to pick the significance level 
    print(outliers)

    # select and output the outliers
    for i in outliers:
        print(f"Outlier {i}")
        plt.figure()
        plt.pcolormesh(read_data[i].astype(np.float64),cmap="binary")
        plt.ylim(0,data_shape[1])
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("Plot Number "+str(i))
        # plt.savefig('/DATA/OCTANS_1/tob023/working/simulations/Benchmarking/plot'+str(i)+'.jpeg', format='jpeg',  dpi=200)
        plt.savefig(f'/u/tob023/working/P505Again/{iterations}Iterations_File{j}plot'+str(i)+'.jpeg', format='jpeg',  dpi=200) # change this path to put all the outlier images wherever you want them to go
        print(f"For file {f}, {i} has error {errors[i]} and zscore {zscores[i]}")
        plt.close()
        plt.clf()

    j = j + 1

print(combined_zscores)
print("--- %s seconds ---" % (time.time() - start_time))

# close psrfits
close_psrfits(hdulist)
