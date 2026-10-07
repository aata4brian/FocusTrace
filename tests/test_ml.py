import copy
import numpy as np
import pytest
from tracefokus.config import load_config
from tracefokus.ml import Normalizer, assert_disjoint, split_groups

def test_assert_disjoint_detects_participant_leakage():
    with pytest.raises(ValueError): assert_disjoint(['P01'],['P02'],['P01'])
    assert_disjoint(['P01'],['P02'],['P03'])

def test_fixed_split_accounts_for_all_participants():
    cfg=load_config(); groups=np.array([f'P{i:02}' for i in range(1,19)])
    train,val,test=next(split_groups(groups,'fixed',cfg))
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert set(train)|set(val)|set(test)==set(groups)

def test_group_kfold_has_zero_leakage():
    cfg=load_config(); cfg=copy.deepcopy(cfg); cfg['evaluation']['folds']=5
    groups=np.repeat(np.array([f'P{i:02}' for i in range(1,19)]),3)
    folds=list(split_groups(groups,'group',cfg))
    assert len(folds)==5
    for train,val,test in folds: assert_disjoint(train,val,test)

def test_normalizer_adds_missingness_mask_without_nan():
    x=np.array([[[1.0,np.nan],[2.0,np.nan]],[[3.0,4.0],[np.nan,5.0]]],dtype=float)
    n=Normalizer().fit(x); z=n.transform(x)
    assert z.shape[-1]==4
    assert np.isfinite(z).all()
