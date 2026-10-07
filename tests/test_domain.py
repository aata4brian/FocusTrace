import pytest
from tracefokus.domain import CODES, PRESETS, Machine, labels, validate_id, validate_sequence

class Clock:
    def __init__(self): self.value=100.0
    def __call__(self): return self.value
    def add(self,seconds): self.value+=seconds

def test_all_presets_have_each_block_once():
    assert len(PRESETS)==18
    for seq in PRESETS.values():
        assert len(seq)==6
        assert set(seq)==set(CODES)

def test_label_mapping_preserves_tree_as_on_task():
    assert labels('A1')==('DTE',0)
    assert labels('B2')==('TREE',0)
    assert labels('C1')==('TIE',1)
    assert labels('TRANSITION')==('TRANSITION',None)

def test_validation_rejects_bad_sequence_and_identifier():
    with pytest.raises(ValueError): validate_sequence(['A1']*6)
    with pytest.raises(ValueError): validate_id('nama peserta')
    assert validate_id('P01')=='P01'

def test_machine_uses_transition_and_server_clock():
    clock=Clock(); m=Machine(PRESETS['P01'],calibration_sec=5,block_sec=10,clock=clock)
    assert m.command('start','start',0)
    assert m.state=='CALIBRATION'
    clock.add(5.2); assert m.advance(); assert m.state=='TRANSITION'
    assert m.intervals[0]['duration']==5
    rev=m.revision
    assert m.command('next','next-1',rev)
    assert m.state=='BLOCK' and m.code=='A1'
    clock.add(10.5); assert m.advance(); assert m.state=='TRANSITION'
    assert m.intervals[-1]['duration']==10

def test_duplicate_command_is_idempotent():
    clock=Clock(); m=Machine(PRESETS['P01'],clock=clock)
    assert m.command('start','same-id',0) is True
    assert m.command('start','same-id',m.revision) is False
