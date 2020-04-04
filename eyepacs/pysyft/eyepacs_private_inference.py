import torch

from common.ResNetCopy import resnext50_32x4d
from eyepacs.common.constants import TEST_CSV_PATH_1
from eyepacs.common.eyepacs_data_loader import EyepacsDataLoader
from eyepacs.pysyft.pysyft_eyepacs import PysyftEyepacs

NUM_CLASSES = 5

pysyft_eyepacs = PysyftEyepacs()
model = resnext50_32x4d()
model.fc = torch.nn.Linear(2048, NUM_CLASSES)

checkpoint = torch.load('../eyepacs/eyepacs_models/arch[resnext50_32x4d]_optim[adam]_criterion[CrossEntropy]_lr[0.0001]_lrsch[cosine]_batch[15]_imsize[512]_WeightedSampling[False]/model_best.pth.tar',
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# data_loader = EyepacsDataLoader('./data/test_private_df.csv', './data/test_private_df.csv')
#
# for data, labels in data_loader.test_loader:
#     out = model(data)
#     pred = out.argmax(dim=1)
#     print(pred)
#     print(labels)


pysyft_eyepacs.encrypt_evaluate_model(model)
