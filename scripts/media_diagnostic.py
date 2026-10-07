r"""Check whether local experiment media is research-ready.

Run from the project root after replacing A2/B2/C1 assets:
    .\.venv\Scripts\python.exe scripts\media_diagnostic.py
"""
from tracefokus.config import load_config
from tracefokus.content import Content


def main() -> int:
    cfg=load_config()
    content=Content(cfg['paths']['content'],cfg.get('content',{}),strict_media=True)
    print('TraceFokus media diagnostic')
    print('===========================')
    for code,info in content.media_info.items():
        if code in ('A2','B2'):
            audio='yes' if info.get('has_audio') else 'no' if info.get('has_audio') is False else 'unknown'
            print(f"{code}: {info.get('duration_sec',0):.1f}s, {info.get('width')}x{info.get('height')}, audio={audio}")
    if content.errors:
        print('\nNOT RESEARCH READY')
        for error in content.errors:print(f'- {error}')
        return 1
    print('\nRESEARCH READY: media validation passed.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
