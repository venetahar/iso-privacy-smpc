from mnist.graphene.json_graphene_rest_mnist_data_loader import JSONGrapheneRESTMnistDataLoader
from common.graphene.rest import graphene_rest_private_inference

import sys


TEST_BATCH_SIZE=100
data_path = sys.argv[1] if len(sys.argv) > 1 else 'mnist/data'

data_loader = JSONGrapheneRESTMnistDataLoader(data_path, TEST_BATCH_SIZE)

graphene_inference = graphene_rest_private_inference.GrapheneRESTClientPrivateInference(
    "http://127.0.0.1:5000",
    data_loader
)

graphene_inference.encrypt_data()
graphene_inference.evaluate()
