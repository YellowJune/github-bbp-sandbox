import argparse,json,math,random,statistics,threading,urllib.parse,urllib.request
from dataclasses import dataclass,asdict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

ACTIONS=['mutate_user_id','mutate_scope','mutate_mode','mutate_page','mutate_role','mutate_token','mutate_limit','mutate_sort']
PROGRAMS=('users','reports','tokens')
USERS={'alice':('alice','A_PRIV'),'bob':('bob','B_PRIV'),'carol':('carol','C_PRIV')}
REPORTS={1:('alice','R1_PRIV'),2:('bob','R2_PRIV'),3:('carol','R3_PRIV'),4:('bob','R4_PRIV')}
TOKENS={'guest-a':('alice','T_A_META'),'guest-b':('bob','T_B_META'),'user-c':('carol','T_C_META')}

def logic(p,q):
    r=q.get('requester','alice'); forbidden=False; sig=None; src='public'
    if p=='users':
        uid=q.get('user_id',r); owner,mark=USERS.get(uid,USERS[r]); forbidden=q.get('scope')=='all' and owner!=r
        sig=f'users:{owner}:http_body' if forbidden else None; src='cross_user' if forbidden else 'public'
    elif p=='reports':
        try: rid=int(q.get('report_id','1'))
        except ValueError: rid=1
        owner,mark=REPORTS.get(rid,REPORTS[1]); forbidden=q.get('mode')=='debug' and q.get('page')=='0' and owner!=r
        sig=f'reports:{rid}:{owner}:http_body' if forbidden else None; src='cross_user' if forbidden else 'public'
    elif p=='tokens':
        tok=q.get('token','guest-a'); owner,mark=TOKENS.get(tok,TOKENS['guest-a']); forbidden=q.get('role')=='admin' and tok.startswith('guest-') and owner!=r
        sig=f'tokens:{owner}:http_body' if forbidden else None; src='auth_private' if forbidden else 'public'
    else: raise KeyError(p)
    body={'status':'ok'}
    if forbidden: body['toy_private_marker']=mark
    return body,{'program':p,'source_domain':src,'sink':'http_body','forbidden':forbidden,'signature':sig}

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        u=urllib.parse.urlparse(self.path); p=u.path.strip('/'); q={k:v[-1] for k,v in urllib.parse.parse_qs(u.query).items()}
        try: body,flow=logic(p,q)
        except KeyError: self.send_response(404); self.end_headers(); return
        b=json.dumps(body,sort_keys=True).encode(); f=urllib.parse.quote(json.dumps(flow,sort_keys=True,separators=(',',':')),safe='')
        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.send_header('X-CLIF-Flow',f); self.end_headers(); self.wfile.write(b)

class Server:
    def __enter__(self):
        self.s=ThreadingHTTPServer(('127.0.0.1',0),H); self.t=threading.Thread(target=self.s.serve_forever,daemon=True); self.t.start(); self.url=f'http://127.0.0.1:{self.s.server_port}'; self.base_url=self.url; return self
    def __exit__(self,*a): self.s.shutdown(); self.s.server_close(); self.t.join(timeout=2)

def http(url,p,q):
    with urllib.request.urlopen(f'{url}/{p}?{urllib.parse.urlencode(q)}',timeout=2) as r: r.read(); return json.loads(urllib.parse.unquote(r.headers['X-CLIF-Flow']))

class RandomPolicy:
    def __init__(self,r): self.r=r
    def choose(self,t): return self.r.choice(ACTIONS)
    def update(self,a,x): pass

class UCB:
    def __init__(self,r,c=1.25): self.r=r; self.c=c; self.n={a:0 for a in ACTIONS}; self.q={a:0.0 for a in ACTIONS}
    def choose(self,t):
        z=[a for a in ACTIONS if not self.n[a]]
        if z: return self.r.choice(z)
        return max(ACTIONS,key=lambda a:self.q[a]+self.c*math.sqrt(math.log(max(2,t))/self.n[a]))
    def update(self,a,x): self.n[a]+=1; self.q[a]+=(x-self.q[a])/self.n[a]

def base(p,r):
    who=r.choice(['alice','bob','carol'])
    if p=='users': return {'requester':who,'user_id':r.choice(['alice','bob','carol']),'scope':'self','limit':'10','sort':'asc'}
    if p=='reports': return {'requester':who,'report_id':str(r.choice([1,2,3,4])),'mode':'normal','page':'1','limit':'10','sort':'asc'}
    return {'requester':who,'token':r.choice(['guest-a','guest-b','user-c']),'role':'user','limit':'10','sort':'asc'}

def mutate(p,q,a,r):
    x=dict(q)
    if a=='mutate_user_id': x['user_id']=r.choice(['alice','bob','carol'])
    elif a=='mutate_scope': x['scope']='all'
    elif a=='mutate_mode': x['mode']='debug'
    elif a=='mutate_page': x['page']='0'; x['mode']='debug' if p=='reports' else x.get('mode','normal')
    elif a=='mutate_role': x['role']='admin'
    elif a=='mutate_token': x['token']=r.choice(['guest-a','guest-b']); x['role']='admin' if p=='tokens' else x.get('role','user')
    elif a=='mutate_limit': x['limit']=str(r.choice([0,1,999]))
    else: x['sort']=r.choice(['asc','desc','none'])
    return x

def sid(s): return sum((i+1)*ord(c) for i,c in enumerate(s))
@dataclass
class Row: program:str; policy:str; trial:int; budget:int; discoveries:int; forbidden_hits:int; first_discovery:int|None; reward_sum:float; top_action:str|None

def trial(url,p,pol,k,budget,seed):
    er=random.Random(seed*1009+k*31+sid(p)*7); pr=random.Random(seed*2027+k*43+sid(pol)*11); policy=UCB(pr) if pol=='ucb' else RandomPolicy(pr); seen=set(); d=h=0; first=None; rew=0.0
    for t in range(1,budget+1):
        a=policy.choose(t); f=http(url,p,mutate(p,base(p,er),a,er)); hit=bool(f['forbidden']); sig=f.get('signature'); new=hit and sig and sig not in seen
        if new: seen.add(sig); d+=1; first=t if first is None else first
        h+=int(hit); x=1.0 if new else (0.2 if hit else 0.0); rew+=x; policy.update(a,x)
    top=max(ACTIONS,key=lambda a:policy.q[a]) if isinstance(policy,UCB) else None
    return Row(p,pol,k,budget,d,h,first,rew,top)

def summary(rows):
    out={'programs':{},'aggregate':{}}
    def s(rr):
        return {'mean_discoveries':statistics.mean(x.discoveries for x in rr),'mean_forbidden_hits':statistics.mean(x.forbidden_hits for x in rr),'mean_first_discovery':statistics.mean(x.first_discovery if x.first_discovery is not None else x.budget+1 for x in rr),'mean_reward':statistics.mean(x.reward_sum for x in rr)}
    for p in PROGRAMS: out['programs'][p]={pol:s([x for x in rows if x.program==p and x.policy==pol]) for pol in ('random','ucb')}
    for pol in ('random','ucb'): out['aggregate'][pol]=s([x for x in rows if x.policy==pol])
    r,u=out['aggregate']['random'],out['aggregate']['ucb']; out['aggregate']['relative']={'discovery_lift_pct':100*(u['mean_discoveries']/r['mean_discoveries']-1),'forbidden_hit_lift_pct':100*(u['mean_forbidden_hits']/r['mean_forbidden_hits']-1),'first_discovery_reduction_pct':100*(1-u['mean_first_discovery']/r['mean_first_discovery']),'reward_lift_pct':100*(u['mean_reward']/r['mean_reward']-1)}; return out

ToyServer=Server
execute_http=http
UCBPolicy=UCB

def main():
    a=argparse.ArgumentParser(); a.add_argument('--trials',type=int,default=40); a.add_argument('--budget',type=int,default=80); a.add_argument('--seed',type=int,default=20260816); a.add_argument('--out',type=Path,default=Path('artifacts/clif_http_mvp_results.json')); z=a.parse_args(); rows=[]
    with Server() as s:
        for p in PROGRAMS:
            for pol in ('random','ucb'):
                for k in range(z.trials): rows.append(trial(s.url,p,pol,k,z.budget,z.seed))
    pay={'mvp':'CLIF HTTP program-local online learner','actual_http_execution':True,'pretraining':False,'external_training_data':False,'cross_program_state_reuse':False,'instrumentation':'X-CLIF-Flow source/sink labels from local toy service','learner':'UCB1-style online mutation-family learner','trials_per_program_policy':z.trials,'execution_budget':z.budget,'seed':z.seed,'summary':summary(rows),'rows':[asdict(x) for x in rows]}; z.out.parent.mkdir(parents=True,exist_ok=True); z.out.write_text(json.dumps(pay,indent=2,sort_keys=True)); print(json.dumps(pay['summary'],indent=2,sort_keys=True))
if __name__=='__main__': main()
