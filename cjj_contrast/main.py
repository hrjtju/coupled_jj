from dataloader import create_dataloaders
from models import ParamPredictor, FeatureExtractor
from utils import seed_everything
import wandb
from train_and_test import train, test

seed_everything(42)

wandb.init(project="cjj-contrast", name="contrast+regression")

train_loader, test_loader = create_dataloaders(batchsize=32)

# 初始化 FeatureExtractor
extractor = FeatureExtractor(in_dim=64, hidden_dim=64, n_layer=5, out_dim=128)

predictor = ParamPredictor(in_dim=128, hid_dim=128, n_params=8)

# 对比学习训练
extractor, predictor = train((extractor, predictor), train_loader, epochs=10, temperature=0.1, log_interval=5)

# 测试评估
results = test((extractor, predictor), test_loader, temperature=0.1)

wandb.finish()