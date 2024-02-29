import hydra
from omegaconf import OmegaConf
from config import DnCNNConfig
from dncnn.utils import get_device
from dncnn.dataset import create_dataloader
#  from dncnn.model import DnCNN


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DnCNNConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()

    #  model = DnCNN(number_of_layers=cfg.params.hidden_count, kernel_size=3).to(device=device)
    #  model.eval()
    data = create_dataloader(
        root_path=cfg.paths.data,
        file_path_noisy=cfg.files.train_data_noisy,
        file_path_sharp=cfg.files.train_data_sharp,
    )


if __name__ == "__main__":
    main()
