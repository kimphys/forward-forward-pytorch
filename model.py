import torch
from torch import nn
from torch.nn import functional as F

from dataset import embedding

class FFLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 num_epochs = 1000, bias=True):
        super(FFLinear, self).__init__()

        self.linear = nn.Linear(in_features = in_features, 
                                out_features = out_features, 
                                bias = bias)
        
        nn.init.xavier_uniform_(self.linear.weight.data, gain=1.0)
        nn.init.zeros_(self.linear.bias.data)

        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = num_epochs

    def forward(self, x):
        x_norm = x.norm(2, 1, keepdim=True)
        x_dir = x / (x_norm + 1e-4)
        res = self.linear(x_dir)
        return self.relu(res)

    def forward_forward(self, x_pos, x_neg):

        for i in range(self.num_epochs):
            x_pos.requires_grad = True
            x_neg.requires_grad = True

            g_pos = torch.mean(torch.pow(self.forward(x_pos), 2), 1)
            g_neg = torch.mean(torch.pow(self.forward(x_neg), 2), 1)

            loss = self.criterion(g_pos, g_neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        with torch.no_grad():
            return self.forward(x_pos), self.forward(x_neg), loss

    def criterion(self, g_pos, g_neg):
        return torch.mean(
            torch.log(
                1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold], 0))
            )
        )

class FFNetwork(nn.Module):
    def __init__(self, in_features=784, num_classes=10):
        super(FFNetwork, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        self.ff_1 = FFLinear(in_features=self.in_features, out_features=512)
        self.ff_2 = FFLinear(in_features=512, out_features=512)
        self.ff_3 = FFLinear(in_features=512, out_features=512)
        self.ff_4 = FFLinear(in_features=512, out_features=512)

        self.layers = [self.ff_1, self.ff_2, self.ff_3, self.ff_4]

    def train(self, data_pos, data_neg):
        h_pos = data_pos.view(-1, self.in_features)
        h_neg = data_neg.view(-1, self.in_features)

        total_loss = 0
        total_cnt = 0

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, FFLinear):
                print(f"Training layer {idx+1} now")
                h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)
                
                total_loss += loss
                total_cnt += 1

            else:
                print(f"Passing layer {idx+1} now")
                x = layer(x)

        print('Loss: ', total_loss / total_cnt)

    def predict(self, data):
        with torch.no_grad():
            goodness_per_label = []
            for cls in range(self.num_classes):
                lbl = torch.tensor([cls] * data.shape[0])
                input = embedding(data, lbl, self.num_classes)

                h = input.view(-1, self.in_features)

                goodness = []
                for layer in self.layers:
                    h = layer(h)
                    goodness += [h.pow(2).mean(1)]
                goodness_per_label += [sum(goodness).unsqueeze(1)]

            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label.argmax(1)