#include <cmath>
#include <cstring>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename T>
void matrix_dot(const std::vector<float> &x, const T &y,
                std::vector<float> &ans, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      ans[j + i * k] = 0;
      for (int e = 0; e < n; e++) {
        ans[j + i * k] += x[i * n + e] * y[j + e * k];
      }
    }
  }
}

void transpose(std::vector<float> &x, int m, int n) {
  float ans[m * n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      ans[j * m + i] = x[i * n + j];
    }
  }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      x[i * n + j] = ans[i * n + j];
    }
  }
}

template <typename T> void print(const T &x, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << x[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n,
                                  const size_t k, float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  /// BEGIN YOUR CODE
  std::vector<float> h(batch * k, 0.0);
  std::vector<float> h_exp(batch * k, 0.0);
  std::vector<float> h_sum(batch, 0.0);
  std::vector<float> Z(batch * k, 0.0);
  std::vector<float> dtheta(n * k, 0.0);
  std::vector<float> batch_X(batch * n, 0.0);
  for (int bi = 0; bi < (m + batch - 1) / batch; bi++) {
    int begin_x = bi * batch;
    int end_x = (bi + 1) * batch;
    if (end_x > m)
      end_x = m;
    int nums = end_x - begin_x;
    for (int i = 0; i < nums * n; i++) {
      if (i + begin_x * n >= m * n) {
        std::cout << i + begin_x * n << std::endl;
      }
      batch_X[i] = X[i + begin_x * n];
    }
    matrix_dot(batch_X, // nums * n
               theta,   // n * k
               h,       // nums * k
               nums, n, k);

    for (int i = 0; i < nums; i++) {
      h_sum[i] = 0;
      // h : nums * k
      for (int j = 0; j < k; j++) {
        h_exp[i * k + j] = std::exp(h[i * k + j]);
        h_sum[i] += h_exp[i * k + j];
      }
    }
    for (int i = 0; i < nums; i++) {
      for (int j = 0; j < k; j++) {
        Z[i * k + j] = h_exp[i * k + j] / h_sum[i]; // nums * k
      }
    }
    for (int i = 0; i < nums; i++) {
      int y_label = y[i + begin_x];
      Z[i * k + y_label] -= 1.0;
    }
    transpose(batch_X, nums, n);
    matrix_dot(batch_X, // n * nums
               Z,       // nums * k
               dtheta, n, nums, k);
    for (int i = 0; i < n * k; i++) {
      dtheta[i] /= nums;
      theta[i] -= lr * dtheta[i];
    }
  }

  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
