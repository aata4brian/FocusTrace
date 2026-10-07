import argparse,json,socket,urllib.request

def main():
    p=argparse.ArgumentParser(description='Local network check and phone URLs');p.add_argument('--port',type=int,default=8000);a=p.parse_args()
    try:ips=sorted({i[4][0] for i in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET)})
    except OSError:ips=['127.0.0.1']
    for ip in ips:
        for page in ('operator','participant','stimulus'):print(f'http://{ip}:{a.port}/{page}')
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/api/health',timeout=3) as r:print('Server:',r.read().decode())
    except OSError:print('Server unreachable. Run python run.py first.');return 1
    print('On Windows: allow Python on PRIVATE networks only. Use ipconfig to confirm Wi-Fi IPv4. Disable AP/client isolation on a trusted hotspot if phones cannot reach laptop. Do not disable the firewall globally.')
    return 0
if __name__=='__main__':raise SystemExit(main())
