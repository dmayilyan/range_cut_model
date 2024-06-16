from wgan.dataset import create_dataloader
from torch import optim
from dncnn.utils import get_device
import hydra
from config import WGANConfig as cfg
from hydra.core.config_store import ConfigStore
from config import WGANConfig

EPOCHS = 100
BATCH_SIE = 2

device = get_device()

cs = ConfigStore.instance()
cs.store(name="config_scheme", node=WGANConfig)


def initial_z():
    Z = torch.Tensor(batch_size, 200, requires_grad=True).normal_(0, 1)
    return Z


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: WGANConfig) -> None:
    train_loader = create_dataloader(
        root_path=cfg.files.root_path,
        file_path_noisy=cfg.files.train_noisy,
        file_path_sharp=cfg.files.train_sharp,
        is_train=True,
    )

    test_loader = create_dataloader(
        root_path=cfg.files.root_path,
        file_path_noisy=cfg.files.train_noisy,
        file_path_sharp=cfg.files.train_sharp,
        is_train=False,
        transform=train_loader.dataset.transform,
    )

    D = Discriminator()
    G = Generator()
    D_optimizer = optim.Adam(D.parameters(), lr=0.00005, betas=(0.5, 0.5))
    G_optimizer = optim.Adam(G.parameters(), lr=0.0025, betas=(0.5, 0.5))

    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    #  mone = mone.cuda()
    #  one = one.cuda()
    D.to(device)
    G.to(device)

    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        for i, data in enumerate(train_loader):
            Z = initial_z().to(device)

            real_labels = torch.ones(batch_size, requires_grad=True).to(device)
            fake_labels = torch.zeros(batch_size, requires_grad=True).to(device)

            D.zero_grad()
            G.zero_grad()

            d_real = d_real.mean()
            d_real.backward(mone)

            fake = G(Z)
            d_fake = D(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one)

            d_loss = d_fake - d_real 
            wasserstein_loss = d_real - d_fake
            D.zero_grad()
            D_optimizer.step()


            fake = G(Z)
            g = D(fake)
            g = g.mean()
            g.backward(mone)
            g_loss = -g

            G_optimizer.step()


        print(f'Epoch{epoch} , D_loss : {d_loss.data[0]:.4}, G_loss : {g_loss.data[0]:.4}')




if __name__ == "__main__":
    main()
