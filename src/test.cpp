#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;

// check address bad access
// g++ -fsanitize=address test.cpp -g -o test
// ./test

template <typename T>
void matrix_dot(const std::vector<float> &x, const T &y,
                std::vector<float> &ans, int m, int n, int k) {
  // std::cerr << k << std::endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      ans[j + i * k] = 0;
      for (int e = 0; e < n; e++) {
        ans[j + i * k] += x[i * n + e] * y[j + e * k];
      }
    }
  }
  // std::cerr << k << std::endl;
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
  // print((float*)X, m, n);
  // print(theta, n, k);
  std::vector<float> h(0.0, batch * k);
  h.resize(batch * k);
  std::vector<float> h_exp(0.0, batch * k);
  h_exp.resize(batch * k);
  std::vector<float> h_sum(0.0, batch);
  h_sum.resize(batch);
  std::vector<float> Z(0.0, batch * k);
  Z.resize(batch * k);
  std::vector<float> dtheta(0.0, n * k);
  dtheta.resize(n * k);
  std::vector<float> batch_X(0.0, batch * n);
  batch_X.resize(batch * n);
  std::cout << h.size() << " " << h_exp.size() << std::endl;
  /// BEGIN YOUR CODE
  for (int bi = 0; bi < (m + batch - 1) / batch; bi++) {
    int begin_x = bi * batch;
    int end_x = (bi + 1) * batch;
    if (end_x > m)
      end_x = m;
    int nums = end_x - begin_x;
    // std::cerr << "beginx " << begin_x << " end x " << end_x << std::endl;
    for (int i = 0; i < nums * n; i++) {
      if (i + begin_x * n >= m * n) {
        std::cout << i + begin_x * n << std::endl;
      }
      batch_X[i] = X[i + begin_x * n];
    }
    // print(batch_X, nums, n);
    // std::cout << "----" << std::endl;
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
    // std::cout << "Z" << std::endl;
    // print(Z, nums, k);
    for (int i = 0; i < nums; i++) {
      int y_label = y[i + begin_x];
      Z[i * k + y_label] -= 1.0;
    }
    // transpose(batch_X, nums, n);
    matrix_dot(batch_X, // n * nums
               Z,       // nums * k
               dtheta, n, nums, k);
    for (int i = 0; i < n * k; i++) {
      dtheta[i] /= nums;
      theta[i] -= lr * dtheta[i];
    }
    // std::cout << "dtheta" << std::endl;
    // print(dtheta, n, k);
  }

  /// END YOUR CODE
}

int main() {
  int m = 50, n = 10, k = 3;
  std::vector<float> X(0, m * n);
  X.resize(m * n);
  for (int i = 0; i < m * n; i++)
    X[i] = 1;
  std::vector<unsigned char> y(0, m);
  y.resize(m);
  for (int i = 0; i < m; i++)
    y[i] = 1;
  std::vector<float> theta(0.0, n * k);
  theta.resize(n * k);
  for (int i = 0; i < n * k; i++)
    theta[i] = 0;
  // print(theta, n, k);
  softmax_regression_epoch_cpp(X.data(), y.data(), theta.data(), m, n, k, 1.0,
                               50);
  print(theta, n, k);
  // print(X, m, n);
  // print(y, m, 1);
  return 0;
}