import pandas as pd
import numpy as np

def train_test_validate_split(dataset, train_split: float, test_split: float, validate_split:float=None, shuffle=True, random_state=None):
    '''
    This function receives a dataset as input and will output a train data set, validation data set (optional) and a test data set

    Parameters:
        dataset : array-like or pd.DataFrame
            The dataset to be split.
        train_split : float
            Fraction of data to allocate for the training set.
        test_split : float
            Fraction of data to allocate for the test set.
        validate_split : float, optional
            Fraction of data to allocate for the validation set. If None, only train and test sets are returned.
        shuffle : bool, default True
            Whether to shuffle the data before splitting.
        random_state : int, optional
            Seed for reproducibility.
    
    Returns:
        If validate_split is provided:
            tuple: (train_set, validation_set, test_set)
        Otherwise:
            tuple: (train_set, test_set)
    '''

    dataset = np.array(dataset)
    n = len(dataset)

    # Seed
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle data
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    # Check if the splits sum to 1
    if validate_split is not None:
        total = train_split + validate_split + test_split
    else:
        total = train_split + test_split

    if abs(total - 1) > 1e-6:
        raise ValueError("The total sum of the splits must be equal to 1.")

    # Split the data
    tr_split_idx = int(train_split*n)

    if validate_split:

        # Get validation index
        val_split_idx = int(validate_split*n + tr_split_idx)

        # Split
        train_set = dataset[indices[:tr_split_idx]]
        val_set = dataset[indices[tr_split_idx:val_split_idx]]
        test_set = dataset[indices[val_split_idx:]]

        return train_set, val_set, test_set
    
    else:

        # Split
        train_set = dataset[indices[:tr_split_idx]]
        test_set = dataset[indices[tr_split_idx:]]

        return train_set, test_set

    
def normalize(data, axis = 0, handle_nan='raise', handle_zero='raise'):
    pass


import numpy as np

def initialize(shape, init_mode: str = 'zero', seed: int = None):
    """
    Initializes an array with the specified shape and mode.

    Parameters:
        shape (tuple or list): Shape of the array to be returned.
        init_mode (str): How to initialize the array. Allowed values are:
            'zero'         -- all zeros,
            'random'       -- small random normal values,
            'xavier_norm'  -- Xavier normal initialization (for 2D shapes),
            'xavier_unif'  -- Xavier uniform initialization (for 2D shapes),
            'he'           -- He initialization (for 2D shapes).
        seed (int, optional): Seed for replicability.

    Returns:
        np.ndarray: An array initialized according to the given mode and shape.
    
    Raises:
        ValueError: When shape is not valid or an unsupported init_mode is provided.
    """

    try:
        shape = tuple(int(x) for x in shape)
    except Exception as e:
        raise ValueError("Shape must be an iterable of integers: " + str(e))
    
    rng = np.random.default_rng(seed)

    if init_mode == 'zero':
        return np.zeros(shape)

    elif init_mode == 'random':
        return rng.normal(loc=0.0, scale=0.01, size=shape)

    elif init_mode in ['xavier_norm', 'xavier']:
        if len(shape) != 2:
            raise ValueError("Xavier initialization requires a 2D shape (fan_in, fan_out).")
        fan_in, fan_out = shape
        scale = np.sqrt(2 / (fan_in + fan_out))
        return rng.normal(loc=0.0, scale=scale, size=shape)

    elif init_mode == 'xavier_unif':
        if len(shape) != 2:
            raise ValueError("Xavier uniform initialization requires a 2D shape (fan_in, fan_out).")
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape)

    elif init_mode == 'he':
        if len(shape) != 2:
            raise ValueError("He initialization requires a 2D shape (fan_in, fan_out).")
        fan_in, _ = shape
        scale = np.sqrt(2 / fan_in)
        return rng.normal(loc=0.0, scale=scale, size=shape)

    else:
        raise ValueError("init_mode must be one of: 'zero', 'random', 'xavier_norm', 'xavier_unif', or 'he'.")

def sort_array(array: np.ndarray) -> np.ndarray:

    '''
    This function sorts an np.ndarray

    Parameters:
        array (np.ndarray): Array to be sorted

    Returns:
        np.ndarray: sorted array
    '''
    if array.ndim == 1:
        sort_idx = np.argsort(array)
    elif array.ndim == 2:
        sort_idx = np.argsort(array[:, 0])
    else:
        raise ValueError("Array must be 1D or 2D")
    return array[sort_idx]
