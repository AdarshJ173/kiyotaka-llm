#!/usr/bin/env python3
"""Minimal local web chat (stdlib only) for the character model.

    python scripts/serve.py --ckpt checkpoints/sft/best.pt --port 8000
    # open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.generate import generate_reply, load_model  # noqa: E402
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kiyotaka</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--fg:#e6e8ee;--muted:#8b93a7;--accent:#7aa2f7;--border:#272b36}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
.wrap{max-width:760px;margin:0 auto;height:100vh;display:flex;flex-direction:column}
header{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}small{color:var(--muted)}
#log{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:86%;padding:10px 13px;border-radius:12px;white-space:pre-wrap}
.user{align-self:flex-end;background:var(--accent);color:#0b1020;border-bottom-right-radius:4px}
.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--border);border-bottom-left-radius:4px}
.bot.typing{color:var(--muted)}
form{display:flex;gap:8px;padding:14px 18px;border-top:1px solid var(--border)}
input[type=text]{flex:1;background:var(--panel);border:1px solid var(--border);border-radius:10px;color:var(--fg);padding:11px 13px;font:inherit;outline:none}
input[type=text]:focus{border-color:var(--accent)}
button{background:var(--accent);color:#0b1020;border:0;border-radius:10px;padding:0 18px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:14px;align-items:center;padding:0 18px 10px;color:var(--muted);font-size:12px}
input[type=range]{accent-color:var(--accent)}
</style></head><body><div class="wrap">
<header><h1>Kiyotaka Ayanokoji</h1><small id="meta"></small></header>
<div id="log"><div class="msg bot">...</div></div>
<div class="row"><label>temperature <span id="tv">0.80</span></label>
<input id="temp" type="range" min="0.2" max="1.3" step="0.05" value="0.8">
<label><input id="st" type="checkbox" checked> stream</label></div>
<form id="f"><input id="q" type="text" autocomplete="off" placeholder="Say something..." autofocus>
<button id="b">Send</button></form>
</div>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),b=document.getElementById('b'),
      temp=document.getElementById('temp'),tv=document.getElementById('tv'),st=document.getElementById('st');
let history=[],busy=false;
temp.oninput=()=>tv.textContent=Number(temp.value).toFixed(2);
function add(role,text){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text||busy)return;
  q.value='';busy=true;b.disabled=true;add('user',text);history.push({role:'user',content:text});
  const box=add('bot','');box.classList.add('typing');
  try{
    if(st.checked){
      const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({history,temperature:Number(temp.value),stream:true})});
      const reader=r.body.getReader(),dec=new TextDecoder();let buf='';
      while(true){
        const {value,done}=await reader.read();if(done)break;
        buf+=dec.decode(value,{stream:true});
        let i;
        while((i=buf.indexOf('\\n\\n'))>=0){
          const ev=buf.slice(0,i);buf=buf.slice(i+2);
          if(!ev.startsWith('data: '))continue;
          try{const j=JSON.parse(ev.slice(6));
            if(j.token){box.textContent+=j.token;box.classList.remove('typing');log.scrollTop=log.scrollHeight}
          }catch(_){}
        }
      }
    }else{
      const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({history,temperature:Number(temp.value)})});
      const j=await r.json();box.textContent=j.reply;box.classList.remove('typing');
    }
    history.push({role:'assistant',content:box.textContent});
  }catch(err){box.textContent='[error] '+err;box.classList.remove('typing')}
  busy=false;b.disabled=false;q.focus();
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    model = None
    tok = None
    retriever = None
    lock = threading.Lock()

    def log_message(self, *args):  # quieter server logs
        pass

    def do_GET(self):
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000:
                self.send_error(413)
                return
            req = json.loads(self.rfile.read(length))
            history = req.get("history", [])
            if not history or history[-1].get("role") != "user":
                raise ValueError("history must end with a user message")
            temperature = float(req.get("temperature", 0.8))
            stream = bool(req.get("stream", False))
        except Exception as exc:  # noqa: BLE001
            self.send_error(400, str(exc))
            return

        memory = None
        examples = None
        if self.retriever is not None:
            hits = self.retriever.search(history[-1]["content"], k=1, threshold=0.55)
            if hits:
                row, sim = hits[0]
                if sim >= 0.72:
                    memory = row["assistant"]
                else:
                    examples = [
                        {"role": "user", "content": row["user"]},
                        {"role": "assistant", "content": row["assistant"]},
                    ]

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(text: str) -> None:
                self.wfile.write(f"data: {json.dumps({'token': text})}\n\n".encode())
                self.wfile.flush()

            if memory:
                emit(memory)
            else:
                with self.lock:
                    generate_reply(
                        self.model, self.tok, history,
                        max_new_tokens=120, temperature=temperature, top_p=0.92,
                        repetition_penalty=1.15, on_text=emit, examples=examples,
                    )
            self.wfile.write(b"data: {\"done\": true}\n\n")
        else:
            if memory:
                reply = memory
            else:
                with self.lock:
                    reply = generate_reply(
                        self.model, self.tok, history,
                        max_new_tokens=120, temperature=temperature, top_p=0.92,
                        repetition_penalty=1.15, examples=examples,
                    )
            body = json.dumps({"reply": reply}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "sft" / "best.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "processed" / "tokenizer.json"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Handler.tok = TokenizerWrapper(args.tokenizer)
    Handler.model, _ = load_model(args.ckpt, device=device)
    index = ROOT / "data" / "processed" / "rag_index.npz"
    if index.exists():
        from src.kiyotaka.rag import Retriever
        Handler.retriever = Retriever(index, device=device)
        print("retrieval memory: on")
    print(f"serving on http://{args.host}:{args.port} ({device})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
