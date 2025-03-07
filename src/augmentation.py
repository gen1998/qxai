from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2


def get_train_transforms(input_shape):
    return Compose(
        [
            Resize(input_shape, input_shape),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_valid_transforms(input_shape):
    return Compose(
        [
            Resize(input_shape, input_shape),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_inference_transforms(input_shape):
    return Compose(
        [
            Resize(input_shape, input_shape),
            ShiftScaleRotate(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
