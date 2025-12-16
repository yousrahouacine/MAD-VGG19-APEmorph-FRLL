from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class MorphDataset(Dataset):
  

    def __init__(self, annotation_file, transform=None, target_transform=None):
        super().__init__()

        # --------------------------------------------------
        # Load dataframe or CSV
        # --------------------------------------------------
        if isinstance(annotation_file, pd.DataFrame):
            df = annotation_file.copy()
        else:
            df = pd.read_csv(str(annotation_file), index_col=False)

        # --------------------------------------------------
        # Handle different column naming conventions
        # --------------------------------------------------
        if {"Path", "Class"}.issubset(df.columns):
            # FRLL / MAD_DB format
            df = df[["Path", "Class"]].rename(
                columns={"Path": "path", "Class": "label"}
            )
        elif {"path", "label"}.issubset(df.columns):
            # APEmorph format
            df = df[["path", "label"]]
        else:
            raise ValueError(
                f"Unsupported annotation format. Found columns: {df.columns.tolist()}"
            )

        # --------------------------------------------------
        # Map labels to integers
        # --------------------------------------------------
        df["label"] = df["label"].replace({
            "bonafide": 0,
            "real": 0,
            "morph": 1
        }).astype(int)

        self.image_paths = df["path"].tolist()
        self.labels = df["label"].tolist()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # Load image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        label = self.labels[index]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
