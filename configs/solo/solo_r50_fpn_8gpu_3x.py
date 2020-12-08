_base_ = [
    "../_base_/datasets/cityscapes_instance.py",
]
# model settings
model = dict(
    type="SOLO",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # C2, C3, C4, C5
        frozen_stages=1,
        style="pytorch",
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
    ),
    bbox_head=dict(
        type="SOLOHead",
        num_classes=81,
        in_channels=256,
        stacked_convs=7,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        with_deform=False,
        loss_ins=dict(type="DiceLoss", use_sigmoid=True, loss_weight=3.0),
        loss_cate=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
    ),
)
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel="gaussian",  # gaussian/linear
    sigma=2.0,
    max_per_img=100,
)
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33],
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 36
device_ids = range(8)
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "./work_dirs/solo_release_r50_fpn_8gpu_3x"
load_from = None
resume_from = None
workflow = [("train", 1)]
