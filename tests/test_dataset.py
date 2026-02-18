from pathlib import Path

from PIL import Image

from src.dataset import get_dataloaders


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (64, 64), color=color)
    image.save(path, format="JPEG")


def test_get_dataloaders_split_and_resize(tmp_path: Path):
    cats_dir = tmp_path / "cats"
    dogs_dir = tmp_path / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(10):
        _create_image(cats_dir / f"cat_{idx}.jpg", (255, 0, 0))
        _create_image(dogs_dir / f"dog_{idx}.jpg", (0, 0, 255))

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=str(tmp_path),
        batch_size=4,
        num_workers=0,
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    assert len(train_loader.dataset) == 16
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 2

    batch_images, batch_labels = next(iter(train_loader))
    assert batch_images.shape[1:] == (3, 224, 224)
    assert batch_labels.ndim == 1
