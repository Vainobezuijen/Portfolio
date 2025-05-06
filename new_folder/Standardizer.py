import numpy as np
import pandas as pd

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data, axis=0, handle_nan='raise', handle_zero_std='raise'):
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


            self.mean = np.mean(data, axis=axis, keepdims=True)
            self.std = np.std(data, axis=axis, keepdims=True)

            if np.any(self.std == 0):
                message = f"Standard deviation is zero along axis {axis} for some elements."
                if handle_zero_std == 'raise':
                    raise ValueError(message)
                elif handle_zero_std == 'warn':
                        import warnings
                        warnings.warn(message)
                # Replace zeros with ones to avoid division by zero.
                else:
                    self.std = np.where(self.std == 0, 1, self.std)

        elif isinstance(data, (pd.Series, pd.DataFrame)):
            if data.isnull().values.any():
                if handle_nan == 'raise':
                    raise ValueError('NaN values in the data')
                elif handle_nan == 'fill':
                    data = data.fillna(data.mean(axis=axis))
            self.mean = data.mean(axis=axis)
            self.std = data.std(axis=axis, ddof=0)
            if (self.std == 0).any():
                message = "Standard deviation is zero for some features/rows."
                if handle_zero_std == 'raise':
                    raise ValueError(message)
                elif handle_zero_std in ['warn', 'ignore']:
                    if handle_zero_std == 'warn':
                        import warnings
                        warnings.warn(message)
                    # Replace zeros with 1. For a DataFrame, use .replace; for Series, also .replace works.
                    self.std = self.std.replace(0, 1)

        else:
            raise TypeError("Unsupported data type: expected np.array, pd.Series, pd.DataFrame, or list.")
    

    def transform(self, data):
        """
        Transforms the data using the stored mean and std.
        
        Parameters:
            data: Array-like or pandas object to transform.
            
        Returns:
            The standardized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted; call fit() first.")
            
        if isinstance(data, list):
            data = np.array(data)
            
        if isinstance(data, np.ndarray):
            return (data - self.mean) / self.std
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return (data - self.mean) / self.std
        else:
            raise TypeError("Unsupported data type: expected np.array, pd.Series, pd.DataFrame, or list.")
    
    def fit_transform(self, data, axis=0, handle_nan='raise', handle_zero_std='raise'):
        """
        Combines fit() and transform() in one call.
        
        Parameters:
            data: Data to fit and then transform.
            
        Returns:
            The standardized data.
        """
        self.fit(data, axis=axis, handle_nan=handle_nan, handle_zero_std=handle_zero_std)
        return self.transform(data)
