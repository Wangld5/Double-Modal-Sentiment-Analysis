import torch
import argparse
from utils import *
from torch.utils.data import DataLoader
from train import MultiModalTrainer
from my_transformers.bert_like import transformer
import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('--dataset', type=str, default='mosei_senti_data_noalign.pkl',
                    help='dataset to use')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--grad_accum', type=int, default=4, help='gradient accumulation')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max gradient')
parser.add_argument('--log_interval', type=int, default=30, help='step to print train information')
parser.add_argument('--save_name', type=str, default='mult', help='name of the best model')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epoch to be trained for')
parser.add_argument('--seed', type=int, default=111, help='seed for training')
args = parser.parse_args()
args.device = torch.device('cuda')
args.n_gpu = torch.cuda.device_count()

# if torch.cuda.is_available():
#     torch.cuda.manual_seed(111)
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     use_cuda = True

print("start loading the data...")
train_data = get_data(args, args.dataset, 'train')
valid_data = get_data(args, args.dataset, 'valid')
test_data = get_data(args, args.dataset, 'test')

load_train_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
load_valid_data = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
load_test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
print('Finish loading the data...')

if __name__ == "__main__":
    model = transformer(args.batch_size, num_layers=6, h=4, d_model=256, ffw_dim=4*256)
    multimodal = MultiModalTrainer(args, logger, model, load_train_data, load_valid_data, load_test_data)
    multimodal.train()

