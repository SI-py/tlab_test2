"""Calendar holdouts for the second KION experiment."""
from dataclasses import dataclass
import pandas as pd
from .data import prepare_data,day_number

@dataclass
class ResearchData:
    train:pd.DataFrame
    rm_valid:pd.DataFrame
    valid:pd.DataFrame
    test:pd.DataFrame
    confirmation:pd.DataFrame
    items:pd.DataFrame
    item_ids:list
    manifest:dict
    config:dict
    @property
    def vocab_size(self):return len(self.item_ids)


def prepare_kion(raw_dir,config):
    effective=dict(config,test_end=config['confirmation_end'])
    prepared=prepare_data(raw_dir,effective)
    primary_end=day_number(config['test_end'])
    test=prepared.test[prepared.test.day<primary_end].reset_index(drop=True)
    confirmation=prepared.test[prepared.test.day>=primary_end].reset_index(drop=True)
    manifest=prepared.manifest.copy()
    manifest['config']=config
    manifest['rows']['test']=len(test);manifest['rows']['confirmation']=len(confirmation)
    manifest['unknown_target_fraction'].update(test=float((test.target==2).mean()),confirmation=float((confirmation.target==2).mean()))
    manifest['version']=2
    manifest['prior_test_reused_as_validation']='2021-08-02..2021-08-08'
    return ResearchData(prepared.train,prepared.rm_valid,prepared.valid,test,confirmation,
                        prepared.items,prepared.item_ids,manifest,config)
