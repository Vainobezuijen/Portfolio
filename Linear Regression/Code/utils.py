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
    
def standardize(data, axis=0, handle_nan='raise', handle_zero_std='raise'):
    '''
    This function standardizes a np.array by subtracting the mean and dividing by the standard deviation
    
    Params
        data: a np.array to standardize
        axis: axis along which to compute the mean and standard deviation (0 for columns, 1 for rows)
        handle_nan: {'raise', 'ignore', 'fill'}
            - 'raise': throw an error if any NaNs are found. -> default setting
            - 'ignore': proceed without handling NaNs.
            - 'fill': fill NaNs with the mean along the specified axis.
        handle_zero_std : {'raise', 'warn', 'ignore'}, default = 'raise'
            - 'raise': throw an error if any standard deviations are zero.
            - 'warn': issue a warning and replace zeros with one to avoid division by zero.
            - 'ignore': silently replace zeros with one.

    Returns
        An np.array in standardized format
    '''

    if isinstance(data, list):
        data = np.array(data)

    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
            if handle_nan == 'raise':
                raise ValueError('NaN values in the data')
            elif handle_nan == 'fill':
                mean_nan = np.nanmean(data, axis=axis, keepdims=True)
                data = np.where(np.isnan(data), mean_nan, data)


        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

        if np.any(std == 0):
            message = f"Standard deviation is zero along axis {axis} for some elements."
            if handle_zero_std == 'raise':
                raise ValueError(message)
            elif handle_zero_std == 'warn':
                    import warnings
                    warnings.warn(message)
            # Replace zeros with ones to avoid division by zero.
            else:
                std = np.where(std == 0, 1, std)

        standardized_data = (data - mean) / std
        return standardized_data

    elif isinstance(data, (pd.Series, pd.DataFrame)):
        if data.isnull().values.any():
            if handle_nan == 'raise':
                raise ValueError('NaN values in the data')
            elif handle_nan == 'fill':
                data = data.fillna(data.mean(axis=axis))
        mean = data.mean(axis=axis)
        std = data.std(axis=axis, ddof=0)
        if (std == 0).any():
            message = "Standard deviation is zero for some features/rows."
            if handle_zero_std == 'raise':
                raise ValueError(message)
            elif handle_zero_std in ['warn', 'ignore']:
                if handle_zero_std == 'warn':
                    import warnings
                    warnings.warn(message)
                # Replace zeros with 1. For a DataFrame, use .replace; for Series, also .replace works.
                std = std.replace(0, 1)
        # Perform the standardization.
        standardized_data = (data - mean) / std
        return standardized_data
    
    else:
        raise TypeError("Unsupported data type: expected np.array, pd.Series, pd.DataFrame, or list.")
    
def normalize(data, axis = 0, handle_nan='raise', handle_zero='raise'):
    pass


def initialize(shape, init_mode: str = 'zero', seed=None):

    '''
    This function initializes an array into the needed form. 

    Params:
        shape: the shape of the array that needs to be returned
        init_mode: how to initialize the array: {'zero', 'random', 'xavier', 'he'}
        seed: for replicability

    Returns: 
        An np.array initialized according to the mode with shape 'shape'
    '''

    if seed is not None:
        np.random.seed(seed)

    if init_mode == 'zero':
        return np.zeros(shape)
    
    elif init_mode == 'random':
        return np.random.randn(*shape) * 0.01
    
    elif init_mode == 'xavier_norm':
        inp, outp = shape
        return np.random.normal(0, np.sqrt(2/(inp+outp)),(inp,outp))
    
    elif init_mode == 'xavier_unif':
        inp, outp = shape
        limit = np.sqrt(6 / (inp + outp))
        return np.random.uniform(-limit, limit, (inp, outp))

    elif init_mode == 'he':
        inp, outp = shape
        return np.random.normal(0, np.sqrt(2/inp), (inp,outp))