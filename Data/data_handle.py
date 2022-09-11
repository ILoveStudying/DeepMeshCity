from Data.data_loader import Dataset_Bousai_CPT, Dataset_Taxibj


def select_data(dataset_cfg, mode):
    if dataset_cfg.data_type == 'CPT':
        dataloader = Dataset_Bousai_CPT(configs=dataset_cfg, mode=mode)
    elif dataset_cfg.data_type == 'taxibj':
        dataloader = Dataset_Taxibj(configs=dataset_cfg, mode=mode)
    return dataloader
