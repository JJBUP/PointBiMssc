_base_ = ["_base_/default_runtime.py",
          "_base_/tests/segmentation.py"]

# misc custom setting
num_worker = 4
batch_size = 12  # bs: total bs in all gpus
batch_size_val = 12  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
batch_size_test = 1
test_set_gpu = 1
save_path = "exp/scannet/"
# weight = "weight/model_best_scv2.pth"  # path to model weight
# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointBiMssc_scv2",
        in_channels=9,
        num_classes=20,
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             ignore_index=-1)
    ]
)

# scheduler settings
epoch = 900
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.02)
scheduler = dict(type="OneCycleLR",
                 max_lr=optimizer["lr"],
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=1000.0)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "./data/scannet"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=["wall", "floor", "cabinet", "bed", "chair",
           "sofa", "table", "door", "window", "bookshelf",
           "picture", "counter", "desk", "curtain", "refrigerator",
           "shower curtain", "toilet", "sink", "bathtub", "other furniture"],
    color_dict={
        'wall': (174, 199, 232),
        'floor': (152, 223, 138),
        'cabinet': (31, 119, 180),
        'bed': (255, 187, 120),
        'chair': (188, 189, 34),
        'sofa': (140, 86, 75),
        'table': (255, 152, 150),
        'door': (214, 39, 40),
        'window': (197, 176, 213),
        'bookshelf': (148, 103, 189),
        'picture': (196, 156, 148),
        'counter': (23, 190, 207),
        'desk': (247, 182, 210),
        'curtain': (66, 188, 102),
        'refrigerator': (219, 219, 141),
        'shower curtain': (140, 57, 197),
        'toilet': (202, 185, 52),
        'sink': (51, 176, 203),
        'bathtub': (200, 54, 131),
        'other furniture': (92, 193, 61)},
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="Voxelize", voxel_size=0.02, hash_type="fnv", mode="train", return_min_coord=True),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "normal"))
        ],
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Voxelize", voxel_size=0.02, hash_type="fnv", mode="train", return_min_coord=True),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "normal"))
        ],
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        save_complete_pred=True,  # false 则保存坐标标签等完整数据，否则仅保存label
        test_cfg=dict(
            voxelize=dict(type="Voxelize",
                          voxel_size=0.02,
                          hash_type="fnv",
                          mode="test",
                          keys=("coord", "color", "normal")
                          ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "index"), feat_keys=("coord", "color", "normal"))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)]
            ]
        )
    ),
)
