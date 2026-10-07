import json
from pathlib import Path

import cv2
import numpy as np

from tracefokus.content import Content


def write_video(path: Path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
    for i in range(5):writer.write(np.full((48,64,3),40+i*10,dtype=np.uint8))
    writer.release()


def manifest():
    feed=[dict(id=f'p{i}',author='a',caption='c',theme='x',image=f'C1/media/{i}.jpg') for i in range(1,13)]
    return {
        'A1':{'title':'A1','questions':[{'id':'q','text':'q'}]},
        'A2':{'title':'A2','video':'A2/video.mp4','questions':[{'id':'q','text':'q'}]},
        'B1':{'title':'B1'},
        'B2':{'title':'B2','video':'B2/video.mp4'},
        'C1':{'title':'C1','feed':feed},
        'C2':{'title':'C2','prompts':['1','2','3','4','5']},
    }


def make_content(tmp_path: Path, photo_factory):
    for folder in ('A2','B2','C1/media'):(tmp_path/folder).mkdir(parents=True,exist_ok=True)
    write_video(tmp_path/'A2/video.mp4');write_video(tmp_path/'B2/video.mp4')
    for i in range(1,13):cv2.imwrite(str(tmp_path/f'C1/media/{i}.jpg'),photo_factory(i))
    (tmp_path/'experiment.json').write_text(json.dumps(manifest()),encoding='utf-8')


def test_research_media_rejects_near_blank_feed_placeholders(tmp_path):
    make_content(tmp_path,lambda i:np.full((900,720,3),238,dtype=np.uint8))
    c=Content(tmp_path,{'require_video_audio':False,'minimum_video_sec':0,'c1_min_posts':12,'c1_min_image_std':8},strict_media=True)
    assert any('placeholder/nyaris kosong' in e for e in c.errors)


def test_research_media_rejects_silent_mp4(tmp_path):
    rng=np.random.default_rng(7)
    make_content(tmp_path,lambda i:rng.integers(0,256,(900,720,3),dtype=np.uint8))
    c=Content(tmp_path,{'require_video_audio':True,'minimum_video_sec':0,'c1_min_posts':12,'c1_min_image_std':8},strict_media=True)
    assert any(e.startswith('A2:') and 'track audio' in e for e in c.errors)
    assert any(e.startswith('B2:') and 'track audio' in e for e in c.errors)


def test_dev_media_can_use_engineering_placeholders(tmp_path):
    make_content(tmp_path,lambda i:np.full((900,720,3),238,dtype=np.uint8))
    c=Content(tmp_path,{},strict_media=False)
    assert c.errors==[]


def test_calibration_reading_is_exposed_only_to_participant(tmp_path):
    rng=np.random.default_rng(11)
    make_content(tmp_path,lambda i:rng.integers(0,256,(900,720,3),dtype=np.uint8))
    data=json.loads((tmp_path/'experiment.json').read_text(encoding='utf-8'))
    data['calibration_screen']={
        'title':'Mengapa Langit Berubah Warna?',
        'instruction':'Bacalah secara wajar.',
        'material':'Paragraf pertama.\n\nParagraf kedua.'
    }
    (tmp_path/'experiment.json').write_text(json.dumps(data),encoding='utf-8')
    c=Content(tmp_path,{},strict_media=False)
    assert c.calibration_public('participant')['title']=='Mengapa Langit Berubah Warna?'
    assert 'Paragraf kedua.' in c.calibration_public('participant')['material']
    assert c.calibration_public('stimulus')=={}
