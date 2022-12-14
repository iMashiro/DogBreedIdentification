
import pandas as pd
import torch

from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.scripts.dataset_process import ProcessDataset
from src.vae import VAE
from src.scripts.dataset import Dataset

class RunVae():
    def __init__(self):
        self.data_path = '../data/'

        self.dataset_processor = ProcessDataset(self.data_path)
        self.vae = VAE()

        self.labels = pd.read_csv(self.data_path + 'labels.csv')
        self.test_labels = pd.read_csv(self.data_path + 'sample_submission.csv')

        self.train_dataset = pd.DataFrame(columns=['tensor'])
        self.test_dataset = pd.DataFrame(columns=['tensor'])

        self.columns = self.test_labels.columns.to_list()
        self.columns.remove('id')

    def generate_tensors(self, labels, dataset, data_type):
        dataset[['tensor'] + self.columns] = labels.apply(self.dataset_processor.image_process, args=[data_type, self.columns], axis=1, result_type='expand')
        self.dataset_processor.image_count = 1

    def train(self, train):
        batch_size = 32
        lr = 0.001
        epochs = 50
        latent_dim = 32

        train_dataset = Dataset(train.train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device selected: ' + str(device))
        model = VAE(latent_dim, batch_size=batch_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        print('Cuda core is available? ' + str(torch.cuda.is_available()))

        x = next(iter(train_loader))

        plt.imshow((x[0].numpy().transpose(1, 2, 0)+1)/2)
        plt.show()

        for epoch in range(1, epochs+1):
            model.train()
            print(f'Epoch {epoch} start')

            for batch_idx, data in enumerate(train_loader):
                print('Batch id: ' + str(batch_idx) + ' of ' + str(len(train_loader)))
                data = data.to(device)
                optimizer.zero_grad()

                recon_batch, mu, logvar = model(data)
                loss = model.loss_function(recon_batch, data, mu, logvar)

                loss.backward()
                optimizer.step()

            model.eval()
            recon_img, _, _ = model(x[:1].to(device))
            img = recon_img.view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)

            plt.imshow((img+1.)/2.)
            plt.show(block=False)
            plt.pause(3)
            plt.close("all")

        torch.save(model.state_dict(), '../model.pt')
        return model

    def run_generation(self, model, train):
        train_dataset = Dataset(train.train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = next(iter(train_loader))

        reconstructed, mu, _ = model(x.to(device))
        reconstructed = [value.view(-1, 3, 64, 64).detach().cpu().numpy().transpose(0, 2, 3, 1) for value in reconstructed]

        first_dog_idx = 0
        second_dog_idx = 4

        plt.imshow((x[first_dog_idx].numpy().transpose(1, 2, 0)+1)/2)
        plt.show(block=False)
        plt.pause(5)
        plt.close("all")

        plt.imshow((x[second_dog_idx].numpy().transpose(1, 2, 0)+1)/2)
        plt.show(block=False)
        plt.pause(5)
        plt.close("all")

        dz = (mu[second_dog_idx] - mu[first_dog_idx]) / 31
        walk = Variable(torch.randn(128, 32, 4, 4)).to(device)
        walk[0] = mu[first_dog_idx]

        for i in range(1, 32):
            walk[i] = walk[i-1] + dz
        walk = model.decoder(walk).detach().cpu().numpy().transpose(0, 2, 3, 1)

        fig = plt.figure(figsize=(25, 16))
        for ii, img in enumerate(walk):
            try:
                ax = fig.add_subplot(4, 8, ii+1, xticks=[], yticks=[])
                plt.imshow((img+1)/2.)
            except:
                continue
        plt.show()

if __name__ == '__main__':
    train = RunVae()
    #train.generate_tensors(train.labels, train.train_dataset, 'train')
    #train.train_dataset.to_pickle('processed_data.pkl')
    #train.generate_tensors(train.test_labels, train.test_dataset, 'test')

    train.train_dataset = pd.read_pickle('../data/processed_data.pkl')
    print('Data loaded.')

    #model = train.train(train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=32, batch_size=32).to(device)
    model.load_state_dict(torch.load('../model.pt'))

    train.run_generation(model, train)




