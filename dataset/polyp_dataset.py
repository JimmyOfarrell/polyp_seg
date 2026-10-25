import json
from pathlib import Path
from monai.data import Dataset
from dataset.polyp_transforms import get_polyp_transforms

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

class PolypDataset(Dataset):
    def __init__(self, split_path, section, num_fold, transforms, dataset_origin=None):
        self.section = section
        self.num_fold = num_fold
        self.dataset_origin = dataset_origin

        data = self._generate_data_list(split_path)
        self.num_samples = len(data)

        super().__init__(data, transform=transforms)

    def __repr__(self):
        return (f"PolypDataset(section='{self.section}', fold={self.num_fold}, num_samples={self.num_samples}, origin={self.dataset_origin})")

    def _generate_data_list(self, split_path):
        with open(split_path) as fp: data = json.load(fp)

        datalist = []
        if self.section == 'train': datalist = data[f'fold{self.num_fold}']['train']
        elif self.section == 'validation': datalist = data[f'fold{self.num_fold}']['val']
        elif self.section == 'test':
            datalist = data['test']
            if self.dataset_origin is not None:
                filtered = [sample for sample in datalist if sample.get("dataset_origin") == self.dataset_origin]
                if not filtered:
                    available = sorted({s.get("dataset_origin") for s in data["test"]})
                    raise ValueError(f"No samples found for dataset_origin='{self.dataset_origin}'. Available origins: {available}")
                datalist = filtered
        else: raise ValueError(f"Unsupported section: '{self.section}'. Available options are ['train', 'validation', 'test'].")

        for sample in datalist:
            sample['image'] = str(PROJECT_ROOT / sample['image'])
            sample['mask']  = str(PROJECT_ROOT / sample['mask'])

        return datalist

def get_polyp_dataset(args, sam_image_dim=None):
    train_transforms, val_transforms = get_polyp_transforms(args, sam_image_dim)
    train_dataset = PolypDataset(split_path=args['split_path'], section='train', num_fold=args['n_fold'], transforms=train_transforms)
    val_dataset = PolypDataset(split_path=args['split_path'], section='validation', num_fold=args['n_fold'], transforms=val_transforms)
    return train_dataset, val_dataset

def get_test_polyp_dataset(args, sam_image_dim=None, dataset_origin=None):
    _, val_test_transforms = get_polyp_transforms(args, sam_image_dim)
    test_dataset = PolypDataset(split_path=args['split_path'], section='test', num_fold=args['n_fold'], transforms=val_test_transforms, dataset_origin=dataset_origin)
    return test_dataset