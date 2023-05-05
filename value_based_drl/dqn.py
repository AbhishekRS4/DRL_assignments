import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNSimple(nn.Module):
    def __init__(self, num_actions):
        super(DQNSimple, self).__init__()
        self.num_actions = num_actions
        self.list_feats = [32, 64, 128]

        self.conv_feature_extractor = nn.Sequential(
            nn.Conv2d(4, self.list_feats[0], kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list_feats[0], self.list_feats[1], kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list_feats[1], self.list_feats[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer = nn.Linear(128, self.num_actions)

    def forward(self, state):
        conv_features = self.conv_feature_extractor(state)

        dense_features = self.avg_pool(conv_features)
        dense_features = torch.flatten(dense_features, 1)

        q_value = self.linear_layer(dense_features)
        return q_value


class DQNResidual(nn.Module):
    def __init__(self, num_actions):
        super(DQNResidual, self).__init__()
        self.num_actions = num_actions
        self.list_feats = [32, 64, 128]

        self.init_conv_layer = nn.Sequential(
            nn.Conv2d(4, self.list_feats[0], kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.residual_conv_block_1 = nn.Conv2d(self.list_feats[0], self.list_feats[1], kernel_size=3, stride=1, padding=2, bias=False)
        self.residual_conv_block_2 = nn.Conv2d(self.list_feats[1], self.list_feats[2], kernel_size=3, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer = nn.Linear(128, self.num_actions)

    def residual_block(self, input_features, num_features):
        residual_features = input_features

        if num_features == self.list_feats[1]:
            output_features = self.residual_conv_block_1(input_features)
            output_features = self.relu(output_features)
            output_features = self.residual_conv_block_1(output_features)
        else:
            output_features = self.residual_conv_block_2(input_features)
            output_features = self.relu(output_features)
            output_features = self.residual_conv_block_2(input_features)

        residual_features += output_features
        residual_features = self.relu(residual_features)

        return residual_features

    def forward(self, state):
        init_conv_features = self.init_conv_layer(state)

        residual_features_1 = self.residual_block(init_conv_features, num_features=self.list_feats[1])
        residual_features_2 = self.residual_block(residual_features, num_features=self.list_feats[2])

        dense_features = self.avg_pool(residual_features_2)
        dense_features = torch.flatten(dense_features, 1)

        q_value = self.linear_layer(dense_features)
        return q_value
