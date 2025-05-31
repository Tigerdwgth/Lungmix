import math
import numpy as np
import torch

def __get__aug__label(a,b):
    """Determines the augmented label based on two input labels.

        a (str): The first label, typically a class name (e.g., 'normal', 'abnormal').
        b (str): The second label, typically a class name (e.g., 'normal', 'abnormal').

        str: 
            - If either `a` or `b` is 'normal', returns the other label.
            - If both `a` and `b` are different and neither is 'normal', returns 'both'.
            - If `a` and `b` are the same and not 'normal', returns that label."""
    if a=='normal':
        return b
    elif b=='normal':
        return a
    elif a!=b:
        return 'both'
    else:
        return a
def __repeat_audio_to_max_length(wav1, wav2):
    """Repeats two audio tensors to the maximum length of the two.
    Args:
        wav1 (torch.Tensor): The first audio tensor.
        wav2 (torch.Tensor): The second audio tensor.
    Returns:
        tuple: A tuple containing:
            - wav1 (torch.Tensor): The first audio tensor repeated to max length.
            - wav2 (torch.Tensor): The second audio tensor repeated to max length.
            - max_length (int): The maximum length of the two audio tensors.
            - combined_audio (torch.Tensor): A tensor initialized to zeros with the max length.
    """
    max_length = max(wav1.shape[1], wav2.shape[1])
    combined_audio = torch.FloatTensor(torch.zeros(1,max_length))
    wav1=wav1.repeat(1,math.ceil(max_length/wav1.shape[1]))[:,:max_length]
    wav2=wav2.repeat(1,math.ceil(max_length/wav2.shape[1]))[:,:max_length]
    return wav1,wav2,max_length,combined_audio

def lungmix(wav1,wav2,label1,label2,smooth=False):
    """Mixes two audio tensors with random shifts and a mask based on loudness.
    Args:
        wav1 (torch.Tensor): The first audio tensor.
        wav2 (torch.Tensor): The second audio tensor.
        label1 (str): The label for the first audio tensor.
        label2 (str): The label for the second audio tensor.
        smooth (bool): Whether to smooth the loudness mask. Default is False.
    Returns:
        tuple: A tuple containing:
            - wav (torch.Tensor): The mixed audio tensor.
            - label (str): The augmented label based on the input labels.
            - timestamp (int): A random timestamp for the audio.
            - lam (float): A random lambda value for mixing.
            - label1 (str): The label for the first audio tensor.
            - label2 (str): The label for the second audio tensor.
            - wav1 (torch.Tensor): The first audio tensor after processing.
            - wav2 (torch.Tensor): The second audio tensor after processing.
            - mask (torch.Tensor): The mask applied to the mixed audio.
            - max_length (int): The maximum length of the two audio tensors.
    """
    wav1, wav2, max_length, combined_audio = __repeat_audio_to_max_length(wav1, wav2)
    shift=int(max_length*np.random.rand())
    wav2=wav2.roll(int(shift),dims=1)
    shift2=int(wav1.shape[1]*np.random.rand())
    lam = np.random.beta(0.5, 0.5)
    wav2=torch.cat([torch.mean(wav2)*torch.ones(1,shift2),wav2],dim=1)
    wav1, wav2, max_length, combined_audio = __repeat_audio_to_max_length(wav1, wav2)
    loudness_mask1=torch.abs(wav1)>torch.abs((torch.mean(wav1)+2*torch.std(wav1)))
    loudness_mask2=torch.abs(wav2)>torch.abs((torch.mean(wav2)+2*torch.std(wav2)))
    loudness_mask=loudness_mask1|loudness_mask2
    if smooth:
        #smooth the mask
        ma_windowsize=201
        loudness_mask=torch.nn.functional.max_pool1d(loudness_mask.float(),
                                                        kernel_size=ma_windowsize,
                                                        stride=1,
                                                        padding=(ma_windowsize-1)//2,
                                                    )
    random_mask = torch.rand(1, max_length) > 0.5
    loudness_mask=loudness_mask*lam
    mask=random_mask+loudness_mask
    mask=mask-(mask>1).float()
    combined_audio[:,:wav1.shape[1]] += wav1[:,:]
    combined_audio=(random_mask)*combined_audio+(~random_mask)*wav2[:,:]
    wav=combined_audio
    label=__get__aug__label(label1, label2)
    return wav,label,lam,label1,label2,wav1,wav2,mask,max_length
