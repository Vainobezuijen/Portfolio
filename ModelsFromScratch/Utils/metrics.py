import numpy as np

class RegressionMetrics:

    @staticmethod
    def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        The MSE measures the average of the squared differences between the predicted values
        and the true values. It is always non-negative, sensitive to outliers, differentiable, 
        convex (not for deep NN's), scale-dependent (Use RMSE).

        Parameters:
            y       : True target values.
            y_pred  : Predicted values.
            
        Returns:
            MSE value.
        """
        y, y_pred = np.asarray(y), np.asarray(y_pred)
        if y.shape != y_pred.shape:
            raise ValueError("Shapes of y and y_pred must match.")
        return 0.5 * np.mean(np.square(y_pred - y))

    @staticmethod
    def mae(y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        It measures the average of the absolute differences between the predicted values and the true values.
        It is non-negative, robust to outliers, non-differentiable (use sub-gradient methods), 
        convex (not for deep NN's).

        Parameters:
            y       : True target values.
            y_pred  : Predicted values.
            
        Returns:
            MAE value.
        """
        y, y_pred = np.asarray(y), np.asarray(y_pred)
        if y.shape != y_pred.shape:
            raise ValueError("Shapes of y and y_pred must match.")
        return np.mean(np.abs(y_pred - y))

    @staticmethod
    def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:

        '''
        The RMSE measures the average deviation of the predictions from the true values. This is sensitive to outliers. Lower RMSE values indicate better model
        performance, representing smaller differences between predicted and actual values. To combat the limitations of RMSE it is normalized by dividing by the 
        standard deviation. The RMSE can be interpreted geometrically as the Euclidean distance between the vector of observed values and the
        vector of predictions, which is analogous to the length of a hypotenuse in a right triangle.

        Parameters:
            y       : True target values.
            y_pred  : Predicted values.

        Returns:
            RMSE value.
        '''

        y, y_pred = np.asarray(y), np.asarray(y_pred)
        if y.shape != y_pred.shape:
            raise ValueError("Shapes of y and y_pred must match.")
        root_mse = np.sqrt(np.mean(np.square(y-y_pred)))
        return root_mse/np.std(root_mse)

    @staticmethod
    def r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
        
        """
        Calculate the R-squared (R^2) coefficient of determination.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            float: The R-squared (R^2) value.
        """
        assert len(y) == len(y_pred), "Input arrays must have the same length."
        mean_y = np.mean(y)
        ss_total = np.sum((y - mean_y) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
class ClassificationMetrics:

    @staticmethod
    def accuracy(y_test:np.ndarray, y_pred:np.ndarray) -> float:
        """
        Calculate the accuracy for a classification task.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            float: Accuracy value
        """
        return np.sum(y_test == y_pred) / len(y_test)
    
    @staticmethod
    def precision(y_test:np.ndarray, y_pred:np.ndarray) -> float:
        """
        Calculate the precision for a classification task.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            float: Precision value
        """
        y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
        if y_test.shape != y_pred.shape:
            raise ValueError("Shapes of y_test and y_pred must match.")
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_test:np.ndarray, y_pred:np.ndarray) -> float:
        """
        Calculate the recall for a classification task.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            float: Recall value
        """
        y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
        if y_test.shape != y_pred.shape:
            raise ValueError("Shapes of y_test and y_pred must match.")
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(y_test:np.ndarray, y_pred:np.ndarray) -> float:
        """
        Calculate the F1 score for a classification task.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            float: F1 score value
        """
        precision = ClassificationMetrics.precision(y_test, y_pred)
        recall = ClassificationMetrics.recall(y_test, y_pred)
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
