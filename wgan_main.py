from wgan.dataset import create_dataloader


def main():
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


if __name__ == "__main__":
    main()
