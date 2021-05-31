""" Machine Learning """

def cross_validate(model, X, y, cv=5):
    from sklearn.model_selection import cross_validate
    # Instanciate model
    model = LinearRegression()
    # 5-Fold Cross validate model
    cv_results = cross_validate(model, X, y, cv=5)
    # Scores
    return cv_results['test_score']
    