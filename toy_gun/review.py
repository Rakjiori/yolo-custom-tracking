import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(r"C:\yolo_gun")
IMG, LBL = ROOT / "photos_jpg", ROOT / "labels_auto"
REVIEWED = ROOT / "reviewed.txt"
PORT = 8765


def load_items():
    if "--all" in sys.argv:
        names = sorted(p.stem for p in IMG.glob("*.jpg"))
    else:
        names = [n for n in (ROOT / "review.txt").read_text(encoding="utf-8").split() if n]
    done = set(REVIEWED.read_text(encoding="utf-8").split()) if REVIEWED.exists() else set()
    items = []
    for n in names:
        t = LBL / f"{n}.txt"
        box = None
        if t.exists():
            parts = t.read_text().split()
            if len(parts) >= 5:
                box = [float(v) for v in parts[1:5]]
        items.append({"name": n, "box": box, "has_label": t.exists(), "done": n in done})
    return items


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>라벨 검수</title>
<style>
 body{margin:0;background:#1e1e1e;color:#eee;font-family:'Malgun Gothic',sans-serif;display:flex;flex-direction:column;height:100vh}
 #bar{padding:8px 14px;background:#2d2d2d;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 #wrap{flex:1;display:flex;justify-content:center;align-items:center;overflow:hidden}
 canvas{cursor:crosshair;max-width:100%;max-height:100%}
 .k{background:#444;border-radius:4px;padding:1px 6px;font-size:13px}
 #st{font-weight:bold} .ok{color:#6c6} .warn{color:#fc6} .bad{color:#f77}
</style></head><body>
<div id="bar"><span id="st"></span><span id="name"></span><span id="info"></span>
<span><span class="k">Enter</span> 맞음/저장 <span class="k">드래그</span> 다시 그리기 <span class="k">Delete</span> 총 없음
<span class="k">←</span><span class="k">→</span> 이동 <span class="k">Z</span> 취소</span></div>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
let items=[], i=0, img=new Image(), draft=null, drag=null;
const c=document.getElementById('c'), g=c.getContext('2d');
fetch('/api/list').then(r=>r.json()).then(d=>{items=d; i=Math.max(0,items.findIndex(x=>!x.done)); show();});
function cur(){return items[i];}
function show(){
  if(!items.length){document.getElementById('st').textContent='검수할 사진이 없습니다';return;}
  draft=null; img=new Image(); img.onload=fit; img.src='/img/'+encodeURIComponent(cur().name)+'.jpg';
}
function fit(){
  const W=document.getElementById('wrap'), s=Math.min(W.clientWidth/img.width, W.clientHeight/img.height);
  c.width=Math.round(img.width*s); c.height=Math.round(img.height*s); draw();
}
window.onresize=()=>items.length&&img.complete&&fit();
function rect(b,col,w){ if(!b)return; const[x,y,bw,bh]=b;
  g.strokeStyle=col; g.lineWidth=w; g.strokeRect((x-bw/2)*c.width,(y-bh/2)*c.height,bw*c.width,bh*c.height);}
function draw(){
  if(!items.length) return;
  g.drawImage(img,0,0,c.width,c.height);
  rect(cur().box, draft?'rgba(0,255,0,.35)':'#0f0', 3); rect(draft,'#ffa500',3);
  const done=items.filter(x=>x.done).length, it=cur();
  document.getElementById('st').textContent=`${i+1} / ${items.length}  (확인 완료 ${done})`;
  document.getElementById('name').textContent=it.name;
  const inf=document.getElementById('info');
  if(it.done){inf.textContent='✔ 확인함'+(it.box?'':' (총 없음)'); inf.className='ok';}
  else if(it.box){inf.textContent='자동 박스 있음'; inf.className='warn';}
  else{inf.textContent='박스 없음 → 드래그로 그려 주세요'; inf.className='bad';}
}
function pos(e){const r=c.getBoundingClientRect(); return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];}
c.onmousedown=e=>{drag=pos(e);};
window.onmousemove=e=>{ if(!drag)return; const p=pos(e).map(v=>Math.min(1,Math.max(0,v)));
  const x1=Math.min(drag[0],p[0]),x2=Math.max(drag[0],p[0]),y1=Math.min(drag[1],p[1]),y2=Math.max(drag[1],p[1]);
  draft=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]; draw();};
window.onmouseup=()=>{ if(drag&&draft&&(draft[2]<0.01||draft[3]<0.01)) draft=null; drag=null; draw();};
function save(box){
  const it=cur();
  fetch('/api/save',{method:'POST',body:JSON.stringify({name:it.name,box:box})}).then(r=>{
    if(!r.ok){alert('저장 실패');return;} it.box=box; it.done=true; next();});
}
function next(){ if(i<items.length-1){i++;show();} else {draw(); alert('마지막 사진입니다. 수고하셨어요!');} }
window.onkeydown=e=>{
  if(e.key==='Enter'){ const b=draft||cur().box; if(!b){alert('박스를 그리거나 Delete(총 없음)를 누르세요');return;} save(b);}
  else if(e.key==='Delete'){ save(null);}
  else if(e.key==='ArrowRight'){ if(i<items.length-1){i++;show();} }
  else if(e.key==='ArrowLeft'){ if(i>0){i--;show();} }
  else if(e.key==='z'||e.key==='Z'){ draft=null; draw(); }
};
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/list":
            self.send(200, json.dumps(load_items()).encode(), "application/json")
        elif self.path.startswith("/img/"):
            f = IMG / Path(unquote(self.path[5:])).name
            if f.exists():
                self.send(200, f.read_bytes(), "image/jpeg")
            else:
                self.send(404, b"not found", "text/plain")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            return self.send(404, b"not found", "text/plain")
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = Path(d["name"]).name
        if not (IMG / f"{name}.jpg").exists():
            return self.send(400, b"bad name", "text/plain")
        box = d.get("box")
        txt = "" if box is None else "0 " + " ".join(f"{float(v):.6f}" for v in box) + "\n"
        (LBL / f"{name}.txt").write_text(txt)
        done = set(REVIEWED.read_text(encoding="utf-8").split()) if REVIEWED.exists() else set()
        done.add(name)
        REVIEWED.write_text("\n".join(sorted(done)), encoding="utf-8")
        self.send(200, b"ok", "text/plain")


if __name__ == "__main__":
    LBL.mkdir(exist_ok=True)
    srv = None
    for port in range(PORT, PORT + 50):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("빈 포트를 찾지 못했습니다.")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"검수 도구 실행 중: {url}  (끝내려면 이 창에서 Ctrl+C)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    srv.serve_forever()
