from torchvision.models import resnext50_32x4d

from eyepacs.pysyft.pysyft_eyepacs import PysyftEyepacs

pysyft_eyepacs = PysyftEyepacs()
model = resnext50_32x4d(pretrained=True)
pysyft_eyepacs.encrypt_evaluate_model(model)
