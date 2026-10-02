import numpy as np
import xarray as xr
import dask.array as da

def split_ALEX(video: xr.DataArray, c1: int = 0, c2: int = 1) -> xr.Dataset:
    '''
    T  - times  [:] 
    C  -      [c1, c2]  !!!!  alternating in a pattern High low Both
    Z  - size   [0]
    Y  - size Y [:]
    X  - size X [:]
    '''
    # split channels 
    channel1All = np.array(video[:,c1,0,:,:])
    channel2All = np.array(video[:,c2,0,:,:])
    times = np.array(video.coords['T'].values)

    # get the ALEX flags as a dictionary
    alex_dict = split_chan1_chan2(channel1All, channel2All, times)

    # append the seperated structures into an xr.Dataset
    y = video.coords['Y'].values
    x = video.coords['X'].values
    data_vars = {}

    data_vars['raw'] = video

    for key, arr in alex_dict.items():
        if key.startswith('times_'):
            continue                                   # times become coordinates below
        exc = key.split('_')[1]                        # 'donor' | 'accept' | 'both'
        tdim = f"T_{exc}"
        data_vars[key] = xr.DataArray(
            arr,
            dims=(tdim, 'Y', 'X'),
            coords={tdim: alex_dict[f"times_{exc}"], 'Y': y, 'X': x},
        )

    # return the appended video
    ds = xr.Dataset(data_vars)
    ds.attrs.update(video.attrs)                       # carry the OME metadata along
    return ds



def split_chan1_chan2(channel1All, channel2All, times) -> dict:
    '''
    Parses channel1, channel2, and timing information and splits based on intensity into ALEX data.
    '''
    assert len(channel1All) == len(channel2All)
    length = len(channel1All)

    # calculate average of each frame in c1
    channel1AverageIntensity = np.mean(channel2All, axis=1)

    #Calculate the average of every 3rd frame, starting on frames 1, 2 and 3
    meanThirdFrame = np.array([np.mean(channel1AverageIntensity[0::3]), np.mean(channel1AverageIntensity[1::3]), np.mean(channel1AverageIntensity[2::3])])

    redFrame = meanThirdFrame.argmin() 

    if redFrame == 0:     # MATLAB case 1
        A = np.arange(0, length, 3)
        D = np.arange(1, length, 3)
        B = np.arange(2, length, 3)
        
    elif redFrame == 1:   # MATLAB case 2
        B = np.arange(0, length, 3)
        A = np.arange(1, length, 3)
        D = np.arange(2, length, 3)
        
    elif redFrame == 2:   # MATLAB case 3
        D = np.arange(0, length, 3)
        B = np.arange(1, length, 3)
        A = np.arange(2, length, 3)

    # Donor Camera
    DD = channel2All[D,:,:]
    DA = channel2All[A,:,:]
    DBoth = channel2All[B,:,:]

    # Acceptor camera
    AD = channel1All[D,:,:]
    AA = channel1All[A,:,:]
    ABoth = channel1All[B,:,:]

    # Flip the acceptor channel videos
    # Acceptor camera
    AD = AD[::-1, ...]
    AA = AA[::-1, ...]
    ABoth = ABoth[::-1, ...]

    # time stamps
    tD = times[D]
    tA = times[A] 
    tBoth = times[B]

    outDict = {
        'donorCam_donor': DD,
        'donorCam_accept': DA,
        'donorCam_both': DBoth,
        'acceptorCam_donor': AD,
        'acceptorCam_accept': AA,
        'acceptorCam_both': ABoth,
        'times_donor': tD,
        'times_accept': tA,
        'times_both': tBoth,
    }

    return outDict