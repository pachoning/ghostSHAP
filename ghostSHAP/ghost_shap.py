from itertools import product
import numpy as np
from math import comb


class GhostShap:
    def __init__(self, predict_fn, data, x_test):
        self.predict_fn = predict_fn
        self.data = data
        self.x_test = x_test
        self.num_individuals = data.shape[0]
        self.num_features = data.shape[1]
        self.inf_value = 1e9

    @staticmethod
    def calculate_comb(p, k):
        num = p - 1
        den = comb(p, k) * k * (p - k)
        return num / den

    def obtain_Z(self, max_length_subset=None):
        total_subsets = 2**self.num_features
        if max_length_subset is None:
            num_individuals_Z = total_subsets
        else:
            num_individuals_Z = max_length_subset
        all_subsets = product([0, 1], repeat=self.num_features)
        idx_individuals = np.random.choice(
            a=total_subsets, size=num_individuals_Z, replace=False
        )
        list_individuals = [
            x for i, x in enumerate(all_subsets) if i in idx_individuals
        ]
        Z = np.array(list_individuals, dtype=int)
        total_ones_Z = np.sum(Z, axis=1)
        idx_empty_set = np.where(total_ones_Z == 0)
        idx_empty_set = idx_empty_set[0]
        contains_empty_set = len(idx_empty_set) > 0
        return Z, contains_empty_set, idx_empty_set

    # Obtain matrix H
    # TO DO: run this in parallel
    def obtain_H(self, Z):
        idx_columns = range(self.num_features)
        ones_vector = np.full(shape=(self.num_individuals, 1), fill_value=1)
        n_rows_Z = Z.shape[0]
        p = self.num_features
        H = np.empty(shape=(n_rows_Z, p))

        for i_z, z in enumerate(Z):
            total_ones = np.sum(z)

            if total_ones < p and total_ones > 0:
                idx_x = np.where(z == 1)
                idx_x = idx_x[0]
                X = self.data[:, idx_x]
                X_intercept = np.concatenate((ones_vector, X), axis=1)
                idx_y = np.where(z == 0)
                idx_y = idx_y[0]
                Y = self.data[:, idx_y]
                beta = np.matmul(
                    np.matmul(
                        np.linalg.inv(np.matmul(X_intercept.T, X_intercept)),
                        X_intercept.T,
                    ),
                    Y,
                )
                x_from_x_test = self.x_test[:, idx_x]
                x_from_x_test_intercept = np.concatenate(([[1]], x_from_x_test), axis=1)
                prediction = np.matmul(x_from_x_test_intercept, beta)
                idx = np.concatenate((idx_x, idx_y))
                h = np.concatenate((x_from_x_test, prediction))
                h_sort = np.reshape(h[np.argsort(idx)], newshape=(1, -1))

            elif total_ones == p:
                h_sort = np.copy(self.x_test)
            else:
                h_sort = H[i_z]

            H[i_z] = h_sort
        return H

    def obtain_Y(self, Z, H, contains_empty_set, index_empty_set, mean_predicted_value):
        n_rows_Z = Z.shape[0]
        p = self.num_features
        Y = np.empty(shape=(n_rows_Z, 1))
        if not contains_empty_set:
            Y = self.predict_fn(H)
        else:
            if index_empty_set == 0:
                Y[0] = mean_predicted_value
                Y[1:] = self.predict_fn(H[1:])
            elif index_empty_set == n_rows_Z - 1:
                Y[:index_empty_set] = self.predict_fn(H[:index_empty_set])
                Y[index_empty_set] = mean_predicted_value
            else:
                Y[:index_empty_set] = self.predict_fn(H[:index_empty_set])
                Y[index_empty_set] = mean_predicted_value
                Y[index_empty_set:] = self.predict_fn(H[index_empty_set:])
        Y = Y - mean_predicted_value
        return Y

    # TO DO: run this in parallel
    def obtain_W(self, Z):
        p = self.num_features
        inf_value = self.inf_value
        S = np.sum(Z, axis=1, dtype=np.int64)
        w = [inf_value if s == 0 or s == p else self.calculate_comb(p, s) for s in S]
        W = np.diag(w)
        return W

    def get_importance(self, max_length_subset=None):
        Z, contains_empty_set, idx_empty_set = self.obtain_Z(
            max_length_subset=max_length_subset
        )
        H = self.obtain_H(Z=Z)
        mean_predicted_value = np.mean(self.predict_fn(self.data))
        Y = self.obtain_Y(
            Z=Z,
            H=H,
            contains_empty_set=contains_empty_set,
            index_empty_set=idx_empty_set,
            mean_predicted_value=mean_predicted_value,
        )
        W = self.obtain_W(Z)
        importance = np.matmul(
            np.matmul(
                np.matmul(np.linalg.inv(np.matmul(np.matmul(Z.T, W), Z)), Z.T), W
            ),
            Y,
        )
        return importance
