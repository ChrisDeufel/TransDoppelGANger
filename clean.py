import os
import shutil
path = "runs/web/RNN/1/checkpoint"

for item in os.listdir(path):
    epoch = "".join([s for s in item if s.isdigit()])
    item_path = "{0}/{1}".format(path, item)
    if os.path.isdir(item_path) and (int(epoch) % 10) != 0:
        shutil.rmtree(item_path)
