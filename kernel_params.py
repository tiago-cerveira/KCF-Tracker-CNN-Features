import numpy as np


class Params:
    def __init__(self):
        self.padding = 1.5                              # extra area surrounding the target
        self.output_sigma_factor = 0.1                  # standard deviation for the desired translation filter output
        self.lmbda = 1e-4                               # regularization weight
        self.learning_rate = 0.02

        self.scale_sigma_factor = 1.0/4                 # standard deviation for the desired scale filter output
        self.number_of_scales = 1                      # number of scale levels
        self.scale_step = 1.02                          # Scale increment factor
        self.scale_model_max_area = 512                 # maximum scale

        self.features = "CNN_TF"
        self.cell_size = 4.0
        self.high_freq_threshold = 2 * 10 ** 66
        self.peak_to_sidelobe_ratio_threshold = 6       # Set to 0 to disable (Detect if the target is lost)
        self.rigid_transformation_estimation = False    # Try to detect camera rotation

        self.visualization = True
        self.debug = False

        self.init_pos = np.array((0, 0))
        self.pos = np.array((0, 0))
        self.target_size = np.array((0, 0))
        self.img_files = None
        self.video_path = None

        self.kernel = Kernel()




# Structure with kernel parameters
class Kernel:
    def __init__(self):
        self.kernel_type = "Linear" # Or Gaussian
        self.kernel_sigma = 0.5