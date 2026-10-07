from copy import deepcopy
from fastapi.testclient import TestClient
from tracefokus.api import create_app
from tracefokus.config import load_config

def cfg_for(tmp_path):
    cfg=deepcopy(load_config())
    cfg['paths']['data']=tmp_path/'data'; cfg['paths']['outputs']=tmp_path/'outputs'; cfg['paths']['models']=tmp_path/'models'
    return cfg

def test_pairing_and_role_scoping(tmp_path):
    app=create_app(cfg_for(tmp_path),dev=True,synthetic=True,launch=False)
    rt=app.state.runtime
    with TestClient(app) as client:
        assert client.get('/api/health').json()['ok'] is True
        assert client.post('/api/pair',json={'role':'operator','pin':'wrong'}).status_code==403
        op=client.post('/api/pair',json={'role':'operator','pin':rt.pin}).json()['token']
        part=client.post('/api/pair',json={'role':'participant','pin':''}).json()['token']
        h={'Authorization':'Bearer '+op}; hp={'Authorization':'Bearer '+part}
        assert client.get('/api/setup',headers=h).status_code==200
        assert client.get('/api/setup',headers=hp).status_code==403
        state=client.get('/api/state',headers=hp).json()
        assert state['phase']=='ready' and state['task_token'] is None
