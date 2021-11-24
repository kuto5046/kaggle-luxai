import os
import torch 
from stable_baselines3 import PPO

from agent_policy import CustomActorCriticCnnPolicy, CustomFeatureExtractor  # pip install stable-baselines3

n_stack = 1
_n_obs_channel = 28
n_obs_channel = _n_obs_channel + 8*(n_stack-1)
features_dim = 128
path = './models/'
filename = 'rl_cnn_model_26400000_steps'

class OnnxablePolicy(torch.nn.Module):
    def __init__(self, feature_extractor, mlp_extractor, action_net, value_net):
        super(OnnxablePolicy, self).__init__()
        self.feature_extractor = feature_extractor
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        features = self.feature_extractor(observation)
        action_hidden, value_hidden = self.mlp_extractor(features)
        return self.action_net(action_hidden), self.value_net(value_hidden)

# convert PPO -> ONNX
model = PPO.load(
    path + f"{filename}.zip", 
    device='cpu', 
    custom_objects={
        'policy_class': CustomActorCriticCnnPolicy,
        'policy_kwargs': dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim)  
)})
onnxable_model = OnnxablePolicy(
    model.policy.features_extractor, 
    model.policy.mlp_extractor, 
    model.policy.action_net,
    model.policy.value_net
    )
dummy_input = torch.randn(1, n_obs_channel, 32, 32)
torch.onnx.export(onnxable_model, dummy_input, path + f"{filename}.onnx", opset_version=9)