import torch
import torch.nn as nn

class SCS_CNN(nn.Module):
    """
    Spatio-Channel Scaling Convolutional Neural Network (SCS-CNN).
    This architecture integrates a learnable scaling mechanism that acts as a
    dual attention module, enhancing both performance and interpretability.
    """
    def __init__(self, num_conv, num_hidd, layer_scale_init_value=0.1, spatial_scale_init_value=0.1):
        super(SCS_CNN, self).__init__()
        
        # --- Convolutional Block 1 ---
        self.pad1 = nn.ZeroPad2d((3, 4, 1, 2))
        self.conv1 = nn.Conv2d(6, num_conv, kernel_size=(4, 8), stride=1)
        self.ssconv1 = nn.Parameter(spatial_scale_init_value * torch.ones((24, 72)))
        self.lsconv1 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)))
        
        # --- Convolutional Block 2 ---
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.pad2 = nn.ZeroPad2d((1, 2, 0, 1))
        self.conv2 = nn.Conv2d(num_conv, num_conv, kernel_size=(2, 4), stride=1)
        self.ssconv2 = nn.Parameter(spatial_scale_init_value * torch.ones((12, 36)))
        self.lsconv2 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)))

        # --- Convolutional Block 3 ---
        self.pad3 = nn.ZeroPad2d((1, 2, 0, 1))
        self.conv3 = nn.Conv2d(num_conv, num_conv, kernel_size=(2, 4), stride=1)
        self.ssconv3 = nn.Parameter(spatial_scale_init_value * torch.ones((6, 18)))
        self.lsconv3 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)))

        # --- Fully Connected Layers ---
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_conv * 6 * 18, num_hidd)
        self.gammalinear = nn.Parameter(layer_scale_init_value * torch.ones((num_hidd,)))
        self.output = nn.Linear(num_hidd, 1)

        # --- Activation ---
        self.tanh = nn.Tanh()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Block 1
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.ssconv1.unsqueeze(0).unsqueeze(0) * x  # Spatial scaling
        x = x * self.lsconv1.view(1, -1, 1, 1)        # Channel scaling
        x = self.tanh(x)
        x = self.maxpool(x)

        # Block 2
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.ssconv2.unsqueeze(0).unsqueeze(0) * x
        x = x * self.lsconv2.view(1, -1, 1, 1)
        x = self.tanh(x)
        x = self.maxpool(x)

        # Block 3
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.ssconv3.unsqueeze(0).unsqueeze(0) * x
        x = x * self.lsconv3.view(1, -1, 1, 1)
        x = self.tanh(x)
        
        # FC Layers
        x = self.flatten(x)
        x = self.linear(x)
        x = self.gammalinear * x
        x = self.tanh(x)
        
        # Output
        output = self.output(x)
        return output.squeeze(-1)
