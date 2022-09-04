import numpy as np


def main():
  batch_size = 2
  input_dim = 4
  output_dim = 6

  x = np.random.rand(batch_size, input_dim)
  ori_wm = np.random.rand(input_dim, output_dim)
  d_wm = np.random.rand(output_dim)
  out_wm = np.random.rand(output_dim, output_dim)

  ord_mult = ori_wm @ np.diag(d_wm) @ out_wm
  ord_out = x @ ord_mult

  print(ord_out)


if __name__ == "__main__":
  main()
