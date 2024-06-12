import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import multiprocessing as mp

from tqdm.auto import tqdm
from matplotlib import pyplot as plot
from typing import Union
from os import path




CPU_COUNT = (len(os.sched_getaffinity(0))
             if 'sched_getaffinity' in dir(os) else mp.cpu_count())

sbox = np.array((
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16), dtype=np.uint8)

def rankKey(data: np.ndarray,
            key: int) -> int:
    """
    Computes the rank of `key` in `data`, i.e.,
    the position of `key` in the sorted array `data`.

    Parameters
    ----------
    `data` :  array_like
        The data to rank.
    `key`  : int
        The key value to rank.

    Returns
    ----------
    The rank of `key` in `data`.
    """

    sort = np.argsort(data)
    rank = np.argwhere(sort == key)
    if rank.shape[0] > 1:
        rank = rank[0]

    return 256 - int(rank[0])

def guessDistance(data: np.ndarray,
                  key: int) -> float:
    """
    Computes the guessing distance of `data` at `key`, i.e.,
    the distance between the correcy key and the second guess.

    Parameters
    ----------
    `data` : array_like
        The data to compute the guessing distance for.
    `key`  : int
        The key value to compute the guessing distance at.

    Returns
    ----------
    The guessing distance of `data` at `key`.
    """

    return (data[key] - np.max(data[data != data[key]])) / (max(data) - min(data))

def guessMetrics(predictions: np.ndarray,
                 key: int) -> tuple[np.ndarray, float]:
    """
    Computes attack metrics, i.e, rank and guessing distance.
    Rank is a list over the number of traces.
    All other metrics can be computed from rank.

    Parameters
    ----------
    `predictions` : array_like
        A numpy array of shape (#traces, #key_values) with the log probabilities of the key guessings.
    `key` : int
        The value of the real key byte to guess.

    Returns
    ----------
    A tuple like `(rank, guessing-distance)`.
    """

    real_key_rank = []
    data = np.cumsum(predictions, axis=0)
    for i in np.arange(data.shape[0]):
        real_key_rank.append(rankKey(data[i, :], key))

    guessing_distance = guessDistance(data[-1, :], key)

    return np.array(real_key_rank), guessing_distance

def sortPredictions(predictions: np.ndarray,
                    intermediate_to_key: np.ndarray) -> np.ndarray:
    """
    Sort the ouptut predictions based on the key value.

    Parameters
    ----------
    `predictions` : array_like, shape=(#traces, #guess_values)
        Predictions to sort.
        Predictions are assumed to be ordered by guessed intermediate value.
    `intermediate_to_key`: array_like, shape=(#traces, #guess_values)
        Mapping from guessed intermediate to key.
        For each trace, a key value is associated to each guessed intermediate value. 

    Returns
    --------
    A copy of `predictions` ordered by key value.

    Raises
    ------
    `AssertionError`
        If there is at least one prediction with only zeroes.
    """
    idx = np.argsort(intermediate_to_key)
    key_bytes_proba = predictions[np.arange(predictions.shape[0])[:, None], idx]
    
    if not np.all(key_bytes_proba > 0):
        # We do not want an -inf here, put a very small epsilon
        zero_predictions = key_bytes_proba[key_bytes_proba != 0]
        assert len(zero_predictions) > 0, \
            "Got a prediction with only zeroes... this should not happen!"
        key_bytes_proba[key_bytes_proba == 0] = np.finfo(predictions.dtype).tiny

    return key_bytes_proba

def attackedKeyByte(key: Union[bytes, list[int]],
                        byte: int) -> int:
        """
        Return the byte of the attacked round key based on the byte.
        
        Parameters
        ----------
        `key` : bytes | list[int]
            Key to attack.
        `byte` : int
            Byte of the key to attack.
        
        
        Returns
        ----------
        The byte of the round key to attack.
        """
        return key[byte]

def invAttackedIntermediate(plains: np.ndarray,
                                intermediates: np.ndarray,
                                byte: int) -> np.ndarray:
        """
        Return the key value used in the attack for each intermediate.
        
        Parameters
        ----------
        `plains` : array_like, shape=(#traces, 16)
            Plaintexts used in the attack.
        `intermediates` : array_like, shape=(#traces,)
            Intermediate values guessed in the attack.
        `byte` : int
            Byte of the key to attack.
        
        Returns
        ----------
        A numpy array of shape (#traces,) with the key guesses.
        """
        keys = []
        key_guess = np.arange(0, 256)
        intermediate_guess = sbox[plains[:, byte, None] ^ key_guess]
        for trace in range(intermediate_guess.shape[0]):
            keys.append(np.where(intermediate_guess[trace] == intermediates[trace])[0])
        return np.array(keys)[:,0]

def plot_ge(rk_avg, traces_per_attack, attack_amount, filename='fig', folder='data'):
    plot.rcParams['figure.figsize'] = (20, 10)
    plot.ylim(-5, 180)
    plot.xlim(0, traces_per_attack + 1)
    plot.grid(True)
    plot.plot(range(1, traces_per_attack + 1), rk_avg, '-')
    plot.xlabel('Number of traces')
    plot.ylabel('Mean rank of correct key guess')

    plot.title(
        f'{filename} Guessing Entropy\nUp to {traces_per_attack:d} traces averaged over {attack_amount:d} attacks',
        loc='center'
    )

    plot.savefig(
        path.normpath(path.join(
            folder, f'{filename}_{traces_per_attack:d}trs_{attack_amount:d}att.svg')),
        format='svg', dpi=1200, bbox_inches='tight'
    )
    plot.close()

# Performs attack
def perform_attacks_per_key(predictions, ptexts, keys, target_byte, n_attacks):
    ranks = []

    key_values = np.random.choice(np.arange(0, 256), n_attacks, replace=False)
    for k in tqdm(key_values, desc='Performing attacks', leave=False):
        key_filter = keys[:, 0] == k
        if np.sum(key_filter) > 0:
            keys_perKey = keys[key_filter]
            predictions_perKey = predictions[key_filter]
            plains_perKey = ptexts[key_filter]

            mapping = [invAttackedIntermediate(plains_perKey, np.array(
                [i]*len(plains_perKey)), target_byte) for i in range(256)]
            
            key_proba = sortPredictions(predictions_perKey, np.array(mapping).T)
            atk_key_byte = attackedKeyByte(keys_perKey[0], target_byte)
            rank_ak, _ = guessMetrics(np.log(key_proba), atk_key_byte)
            ranks.append(rank_ak - 1)
    return ranks