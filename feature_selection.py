"""Feature selction preprocessor."""
import pickle

from sklearn.feature_selection import f_regression

if __name__ == '__main__':
    X_vector_path = 'model_base_data/array_cache/X_array_0.npz'
    with open(X_vector_path, 'rb') as X_vector_file:
        X_vector = pickle.load(X_vector_file)['arr_0']
    y_vector_path = 'model_base_data/array_cache/y_array_0.npz'
    with open(y_vector_path, 'rb') as y_vector_file:
        y_vector = pickle.load(y_vector_file)['arr_0']

    f_reg = f_regression(X, y)
    print(f_reg)