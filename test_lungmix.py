import torch
from lungmix import __get__aug__label, __repeat_audio_to_max_length, lungmix

def test_get_aug_label():
    assert __get__aug__label('normal', 'abnormal') == 'abnormal'
    assert __get__aug__label('abnormal', 'normal') == 'abnormal'
    assert __get__aug__label('abnormal', 'critical') == 'both'
    assert __get__aug__label('critical', 'critical') == 'critical'

def test_repeat_audio_to_max_length():
    wav1 = torch.ones(1, 5)
    wav2 = torch.ones(1, 10)
    result = __repeat_audio_to_max_length(wav1, wav2)
    assert result[0].shape[1] == 10
    assert result[1].shape[1] == 10
    assert result[2] == 10

def test_lungmix():
    wav1 = torch.ones(1, 50)
    wav2 = torch.ones(1, 100)
    label1 = 'normal'
    label2 = 'abnormal'
    result = lungmix(wav1, wav2, label1, label2)
    assert result[1] == 'abnormal'
    assert result[5].shape == result[6].shape