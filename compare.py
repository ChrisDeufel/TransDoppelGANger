# imports
import os
import logging
import pandas as pd
import numpy as np

dataset = "FCC_MBA"
# SET UP LOGGING
logging_dir = "evaluation/compare/{}".format(dataset)
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
logging_file = "{}/mesa_meta_corr.log".format(logging_dir)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(logging_file)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# define model directories
transformer_path = "evaluation/FCC_MBA/TRANSFORMER"
RNN_path = "evaluation/FCC_MBA/RNN"
# RNN_original_path = "evaluation/FCC_MBA/RNN_ORIGINAL"
eval_dirs = [transformer_path, RNN_path]
overall_best_result = float('inf')
overall_best_result_dir = ""
for eval_dir in eval_dirs:
    for item in os.listdir(eval_dir):
        sub_dir = "{}/{}".format(eval_dir, item)
        logger.info("MODEL TO EVALUATE: {}".format(sub_dir))
        sub_dir = "{}/meta_meas_correlation".format(sub_dir)
        # walk through files of current model
        best_result_sdir = float('inf')
        best_result_sdir_epoch = ""
        for item in os.listdir(sub_dir):
            if os.path.isfile("{}/{}".format(sub_dir, item)):
                epoch = "".join([s for s in item if s.isdigit()])
                # load csv file
                file_path = "{}/{}".format(sub_dir, item)
                df = np.genfromtxt(file_path, delimiter=';', encoding='utf-8')
                # avg_value = df['Total average'].values[0]
                avg_value = df[1, -1]
                logger.info("EPOCH {} average meta meas correlation: {}".format(epoch, avg_value))
                if avg_value < best_result_sdir:
                    best_result_sdir = avg_value
                    best_result_sdir_epoch = epoch
        logger.info("BEST RESULT: {} EPOCH: {}\n".format(best_result_sdir, best_result_sdir_epoch))
        if best_result_sdir < overall_best_result:
            overall_best_result = best_result_sdir
            overall_best_result_dir = "/".join(sub_dir.split("/")[:-1])
logger.info("OVERALL BEST RESULT: {} MODEL: {}".format(overall_best_result, overall_best_result_dir))
