from train_factor_regularized import train_factor_regularized_keras

backbone_names=["efficientnetb0","densenet121", "mobilenet"]#

for backbone in backbone_names:
    train_factor_regularized_keras(backbone_name = backbone)
