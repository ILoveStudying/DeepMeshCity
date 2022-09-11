# data

device = 'cuda:0'  # cuda:0
random_seed = 6666

# file hyperparameter
StartDate = '20170401'
EndDate = '20170709'
CITY = 'osaka'
DataPath = './Data/osaka/density/'
DataFile = DataPath + 'density_osaka_{}_{}_30min.npy'.format(StartDate, EndDate)
DateFile = DataPath + '/day_information_onehot.csv'

# data hyperparameter
INTERVAL = 30
TimeStep = 6
DaytimeStep = int(24 * 60 / INTERVAL)
trainRatio = 0.8
validRatio = 0.2
MAX_FLOWIO = 1814.0
periodic_len = 1
trend_len = 1

# model hyperparameter
batch_size = 4
width = 60
ratio = 2
patch_size = width // ratio
channel = 1
num_hidden = [64, 64]
meta_date_dim = 57

dataset_cfg = dict(
    dataset_name=CITY,
    type='density',
    data_type='CPT',
    timestep=TimeStep,
    max_value=MAX_FLOWIO,
    daytimestep=DaytimeStep,
    perioid_len=periodic_len,
    trend_len=trend_len,
    trainRatio=trainRatio,
    validRatio=validRatio,
    datapath=DataPath,
    datafile=DataFile,
    datepath=DateFile,
    DateStart=StartDate,
    DateEnd=EndDate,
    freq='30min',
    train_data_paths=DataPath + 'train_osaka_density.npz',
    valid_data_paths=DataPath + 'valid_osaka_density.npz',
    test_data_paths=DataPath + 'test_osaka_density.npz',
)

# model parameter
model_cfg = dict(
    model_type='DeepMeshCity',
    meta_dim=meta_date_dim,
    is_metadate=True,
    map_width=width,
    input_length=TimeStep,
    patch_size=patch_size,
    map_channel=channel,
    num_hidden=num_hidden,
    num_layers=len(num_hidden),
    filter_size=3,
    stride=1,
    padding=1,
    layer_norm=1,
    device=device,
)

train_cfg = dict(
    batch_size=batch_size,
    input_length=TimeStep,
    num_epochs=155,
    max_value=MAX_FLOWIO,
    test_interval=1,
    num_save_samples=3,
    # optimizer_cfg=dict(type='adam', lr=1e-3),
    optimizer_cfg=dict(type='adamw', lr=1e-3, weight_decay=5e-4),
    lr_scheduler_cfg=dict(policy='cos', T_0=5, T_mult=2, eta_min=1e-5),
    check_point_file='checkpoint/osaka/density/DeepMeshCity.pth',
    gen_frm_dir='Results/osaka/density/DeepMeshCity'
)

test_cfg = dict(
    batch_size=batch_size,
    input_length=TimeStep,
    max_value=MAX_FLOWIO,
    num_save_samples=10,
    test_interval=5,
    check_point_file='checkpoint/osaka/density/DeepMeshCity.pth',
    gen_frm_dir='Results/osaka/density/DeepMeshCity'
)
