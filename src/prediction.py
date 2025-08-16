

# calculating predictions
def get_predictions(model,X_test):
    """
    get predicted labels
    Args: 
        model (sklearn.linear_model): statement column  
        X_test (pandas.core.frame.DataFrame): test data
    Returns:
        prediction
    """
    pred = model.predict(X_test)
    
    return pred