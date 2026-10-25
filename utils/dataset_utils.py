from torch.utils.data import DataLoader
from dataset.polyp_dataset import get_polyp_dataset, get_test_polyp_dataset

TEST_DATASETS = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB"]

def get_train_val_dataloaders(args, img_dim):
    batch_size = int(args.get('Batch_size', 4))
    num_workers_train = int(args.get('nW', 0))
    num_workers_val   = int(args.get('nW_eval', 0))
    pin_memory = 'cuda' in str(args.get('device', 'cpu')).lower()

    train_dataset, val_dataset = get_polyp_dataset(args, sam_image_dim=img_dim)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train, drop_last=True, pin_memory=pin_memory, persistent_workers=(num_workers_train > 0))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers_val, drop_last=False, pin_memory=pin_memory, persistent_workers=(num_workers_val > 0))

    print('\n'+'='*60)
    print(f"[DATALOADER] Train/Val loaders created")
    print(f"[DATALOADER] Train samples: {len(train_dataset)}, Batch size: {batch_size}")
    print(f"[DATALOADER] Train batches per epoch: {len(train_dataloader)}")
    print(f"[DATALOADER] Val samples: {len(val_dataset)}")
    print(f"[DATALOADER] Workers - Train: {num_workers_train}, Val: {num_workers_val}")
    print(f"[DATALOADER] Pin memory: {pin_memory}")
    print('='*60)
    return train_dataloader, val_dataloader

def get_test_dataloader(args, img_dim, dataset_origin):
    num_workers_test = int(args.get('nW_eval', 0))
    pin_memory = 'cuda' in str(args.get('device', 'cpu')).lower()

    test_dataset = get_test_polyp_dataset(args, sam_image_dim=img_dim, dataset_origin=dataset_origin)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers_test, drop_last=False, pin_memory=pin_memory, persistent_workers=(num_workers_test > 0))

    print('\n'+'='*60)
    print(f"[DATALOADER] Test loader created")
    print(f"[DATALOADER] Dataset origin: {dataset_origin}")
    print(f"[DATALOADER] Test samples: {len(test_dataset)}")
    print('='*60)

    return test_loader

def get_all_test_dataloaders(args, img_dim):

    test_loaders = {name: get_test_dataloader(args, img_dim, dataset_origin=name) for name in TEST_DATASETS}

    print('\n'+'='*60)
    print(f"[DATALOADER] Summary of all test loaders:")
    for name, loader in test_loaders.items(): print(f"[DATALOADER]   {name}: {len(loader.dataset)} samples")
    print('='*60)

    return test_loaders