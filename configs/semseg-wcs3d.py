_base_ = ["_base_/default_runtime.py",
          "_base_/tests/segmentation.py"]

# misc custom setting
num_worker = 4
batch_size = 9  # bs: total bs in all gpus
batch_size_val = 9  # auto adapt to bs 1 for each gpu
mix_prob = 0.8
empty_cache = False
enable_amp = True
batch_size_test = 4
test_set_gpu = 1
# weight = "weight/model_best_wcs3d.pth"  # path to model weight
save_path = "log/wcs3d"
# resume = True


# dataset settings
dataset_type = "WCS3DDataset"
work_key_word_list = ["Work_1", "Work_2", "Work_3"],  # todo
data_root = "./data/wcs3d"

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointBiMssc_wcs3d",
        in_channels=6,
        num_classes=15,
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             ignore_index=-1)
    ]
)

# scheduler settings
epoch = 3600
eval_epoch = 120
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.02)
scheduler = dict(type="OneCycleLR",
                 max_lr=optimizer["lr"],
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=1000.0)


data = dict(
    num_classes=15,
    ignore_index=-1,
    names=[
        'Shed',
        'Concretehouse',  # 居民地
        'Cementroad',
        'Dirtroad',  # 交通
        'Slope',
        'Scarp',
        'Dam',  # 地貌
        # 'Electrictower',  # 管线
        'Vegetablefield',
        'Grassland',
        'Dryland',
        'Woodland',
        'Bareland',  # 植被与土质
        'Waterline',
        'Ditch',  # 水系
        'Others'  # 其他
    ],
    color_dict={
        'Shed': [0, 191, 255],
        'Concretehouse': [255, 99, 71],
        'Cementroad': [112, 128, 144],
        'Dirtroad': [210, 180, 140],
        'Slope': [199, 21, 133],
        'Scarp': [240, 128, 128],
        'Dam': [123, 104, 238],
        'Vegetablefield': [107, 142, 35],
        'Grassland': [173, 255, 47],
        'Dryland': [255, 165, 0],
        'Woodland': [34, 139, 34],
        'Bareland': [255, 215, 0],
        'Waterline': [70, 130, 180],
        'Ditch': [160, 82, 45],
        'Others': [211, 211, 211]
    },
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="Voxelize", voxel_size=0.25, hash_type="fnv", mode="train",
                 keys=("coord", "color", "segment"), return_discrete_coord=True),
            dict(type="SphereCrop", point_max=200000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "segment"), feat_keys=["coord", "color"])
        ],
        test_mode=False
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="Copy", keys_dict={"coord": "origin_coord", "segment": "origin_label"}),
            dict(type="Voxelize", voxel_size=0.25, hash_type="fnv", mode="train",
                 keys=("coord", "color", "segment"), return_discrete_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 # keys=("coord", "discrete_coord", "segment"),  # 没用discrete_coord 不使用有影响吗？
                 keys=("coord", "segment"),  # 没用discrete_coord 不使用有影响吗？
                 offset_keys_dict=dict(offset="coord"),
                 feat_keys=["coord", "color"])
        ],
        test_mode=False),

    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor")
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="Voxelize",
                voxel_size=0.25,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_discrete_coord=False),
            # crop=dict(type="SphereCrop", point_max=1200000, mode="all"),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "color"))
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
