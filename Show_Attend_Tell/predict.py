import matplotlib.pyplot as plt
import pickle
from core2.solver import CaptioningSolver
from core2.model import CaptionGenerator
from core2.utils import load_coco_data
from core2.bleu import evaluate

plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data = load_coco_data(data_path='./data', split='val')
with open('./data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                           dim_hidden=1024, n_time_step=16, prev2out=True,
                           ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

solver = CaptioningSolver(model, data, data, n_epochs=15, batch_size=128, update_rule='adam',
                            learning_rate=0.0025, print_every=2000, save_every=1, image_path='./image/val2014_resized',
                            pretrained_model=None, model_path='./model/lstm', test_model='./model/lstm/model-40',
                            print_bleu=False, log_path='./log/')

solver.test(data, split='val')
