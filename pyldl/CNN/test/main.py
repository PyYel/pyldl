
import os, sys



MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__)) # saves model wieghts to /test folder by default
    sys.path.append(MAIN_DIR)

from CNN import CNNDetectionSSD


model = CNNDetectionSSD(version="320")
model._repair_model()
