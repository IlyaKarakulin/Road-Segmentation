from model import Segmentator

path_to_model = './meta_data/models/best.pth'
path_to_test = './data/test'

model = Segmentator('cuda:0')

model.load_model(path_to_model)

model.test(path_to_test, 4)