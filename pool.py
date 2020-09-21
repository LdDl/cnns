from torch import nn
from torch import torch

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
        self.relu1 = nn.ReLU()
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(9, 3, bias=False)

        conv_weights = torch.Tensor(
            [
                [
                    [
                        [0.10466029,	-0.06228581,	-0.43436298],
                        [0.44050909,	-0.0753625,	-0.34348075],
                        [0.16456005,	0.18682307,	-0.40303048]
                    ]
                ]
            ]
        )
        fc_weights = torch.Tensor(
            [
                [-0.19908814, 0.01521263, 0.31363996, -0.28573613, -0.11934281, -0.18194183, -0.03111016, -0.21696585, -0.20689814],
                [0.17908468, -0.28144695, -0.29681312, -0.13912858, 0.07067328, 0.36249144, -0.20688576, -0.20291744, 0.25257304],
                [-0.29341734, 0.36533501, 0.19671917, 0.02382031, -0.47169692, -0.34167172, 0.10725344, 0.47524162, -0.42054638],
            ]
        )

        with torch.no_grad():
            self.conv1.weight.data = conv_weights
            self.fc1.weight.data = fc_weights

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool2d1(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

def my_loss(output, target):
    loss = output - target
    return torch.FloatTensor(loss)

image = torch.tensor(
    [
        [
            [
                [-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [-0.9, -0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
                [-0.17, 0.18, -0.19, 0.20, 0.21, 0.22, 0.23, 0.24],
                [-0.25, 0.26, 0.27, -0.28, 0.29, 0.30, 0.31, 0.32],
                [-0.33, 0.34, 0.35, 0.36, -0.37, 0.38, 0.39, 0.40],
                [-0.41, 0.42, 0.43, 0.44, 0.45, -0.46, 0.47, 0.48],
                [-0.49, 0.50, 0.51, 0.52, 0.53, 0.54, -0.55, 0.56],
                [-0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, -0.64],
                [-0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72]
            ]
        ]
    ]
)

desired = torch.Tensor(
    [
        [0.32, 0.45, 0.96]
    ]
)

model = ConvNet()
print("Model", model)
print("Desired", desired)
predicted = model(image)
print("Got", predicted)