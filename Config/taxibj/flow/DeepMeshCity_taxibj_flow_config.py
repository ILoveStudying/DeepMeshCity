# data

device = 'cuda:0'  # cuda:0
random_seed = 6666

# file hyperparameter
DataPath = './Data/taxibj/flow/'

# data hyperparameter
INTERVAL = 30
TimeStep = 6
DaytimeStep = int(24 * 60 / INTERVAL)
MAX_FLOWIO = 1292.0
periodic_len = 1
trend_len = 1

# model hyperparameter
batch_size = 32
width = 32
ratio = 4
patch_size = width // ratio
channel = 2
num_hidden = [64, 64]
meta_date_dim = 28

dataset_cfg = dict(
    data_type='taxibj',
    max_value=MAX_FLOWIO,
    datapath=DataPath,
)

# model parameter
model_cfg = dict(
    model_type='DeepMeshCity_TaxiBJ',
    meta_dim=meta_date_dim,
    is_metadate=True,
    map_width=width,
    clossness_len=TimeStep,
    periodic_len=periodic_len,
    trend_len=trend_len,
    patch_size=patch_size,
    map_channel=channel,
    num_hidden=num_hidden,
    num_layers=len(num_hidden),
    filter_size=3,
    stride=1,
    layer_norm=1,
    device=device,
)

train_cfg = dict(
    batch_size=batch_size,
    input_length=TimeStep,
    num_epochs=75,
    max_value=MAX_FLOWIO,
    test_interval=1,
    num_save_samples=3,
    # optimizer_cfg=dict(type='adam', lr=1e-3),
    # lr_scheduler_cfg=None,
    optimizer_cfg=dict(type='adamw', lr=1e-3, weight_decay=5e-4),
    lr_scheduler_cfg=dict(policy='cos', T_0=5, T_mult=2, eta_min=1e-5),
    check_point_file='checkpoint/taxibj/flow/DeepMeshCity.pth',
    gen_frm_dir='Results/taxibj/flow/DeepMeshCity'
)

test_cfg = dict(
    batch_size=batch_size,
    input_length=TimeStep,
    max_value=MAX_FLOWIO,
    num_save_samples=10,
    test_interval=5,
    check_point_file='checkpoint/taxibj/flow/DeepMeshCity.pth',
    gen_frm_dir='Results/taxibj/flow/DeepMeshCity'
)
