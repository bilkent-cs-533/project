import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

feature_size = 46088
reduced_feature_size1 = 10000
reduced_feature_size2 = 900

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, reduced_feature_size2))
        self.decoder = nn.Sequential(
            nn.Linear(reduced_feature_size2, feature_size), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_autoencoder(features, num_epochs = 10):
    model = autoencoder().cuda(0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    batch_size = 150
    for epoch in range(num_epochs):
        for i in range(0, len(features), batch_size):
            sample = features[i:i+batch_size]
            if i % (batch_size*5) == 0: 
                print(i, end = "\r")
            img = Variable(torch.from_numpy(sample.astype(np.float32))).cuda(0)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
        """
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))
        """
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), './sim_autoencoder.pth')

def reduce_features(features):
    state_dict = torch.load('./sim_autoencoder.pth')
    weights = state_dict["encoder.0.weight"]
    biases = state_dict["encoder.0.bias"]

    z = np.ones((len(features), 1), dtype=type(features[0][0]))
    features = np.append(features, z, axis=1)
    weights = np.transpose(weights.cpu().numpy())
    biases = [biases.cpu().numpy()]
    weights = np.append(weights, biases, axis = 0)

    new_features = np.dot(features, weights)
    return new_features