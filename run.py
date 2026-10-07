"""Single-process production launcher; no npm or internet needed after installation."""
import argparse,os,socket,sys,webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
def local_ips():
    ips={'127.0.0.1'}
    try:ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET))
    except OSError:ips.add('127.0.0.1')
    return sorted(ips)
def main():
    p=argparse.ArgumentParser(description='TraceFokus v2 local research server')
    p.add_argument('--dev',action='store_true',help='5s calibration, 10s blocks, separate outputs')
    p.add_argument('--synthetic-camera',action='store_true',help='Engineering-only camera; requires --dev')
    p.add_argument('--host');p.add_argument('--port',type=int);p.add_argument('--open',action='store_true',help='Open participant page')
    args=p.parse_args()
    try:
        import uvicorn
        from tracefokus.config import load_config
        from tracefokus.api import create_app
        cfg=load_config(ROOT)
        host=args.host or cfg['server']['host'];port=args.port or cfg['server']['port']
        app=create_app(cfg,args.dev,args.synthetic_camera)
        if not (ROOT/'frontend/dist/index.html').exists():raise ValueError('Frontend dist missing. Run npm.cmd ci && npm.cmd run build inside frontend.')
        print('\nTraceFokus v2 | LOCAL RESEARCH PLATFORM',flush=True)
        print('DEV MODE — NOT RESEARCH DATA' if args.dev else 'Research timing: 30s calibration + 6 x 120s blocks',flush=True)
        print('Operator PIN: '+app.state.runtime.pin,flush=True)
        for ip in local_ips():
            for route in ['operator','participant','stimulus','monitor','review','content-preview','demo']:print(f'http://{ip}:{port}/{route}',flush=True)
        print('Private Wi-Fi/hotspot only. Do not expose this HTTP service to the internet.',flush=True)
        if args.open:webbrowser.open(f'http://127.0.0.1:{port}/participant')
        uvicorn.run(app,host=host,port=port,access_log=False)
    except (ImportError,ValueError,OSError) as exc:
        print(f'Startup failed: {exc}\nSee docs/TROUBLESHOOTING.md.',file=sys.stderr);return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
