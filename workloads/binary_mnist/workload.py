from workloads.spec import Workload


class BinaryMnistWorkload(Workload):

  @property
  def batch_size(self):
    return 128

  @property
  def num_epochs(self):
    return 500

  @property
  def latent_dim(self):
    return 32

  @property
  def reconsturction_loss(self):
    return "bce"

  
