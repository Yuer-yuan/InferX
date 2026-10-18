"""Shared native-object figures for the TeX decks and editable draw.io files."""
from xml.etree import ElementTree as ET

COLORS = {
    'neutral': ('CCD3DB', 'FFFFFF'),
    'mechanism': ('7E2635', 'F9F2F3'),
    'state': ('32649A', 'F2F6FA'),
    'accept': ('7E2635', 'F0DEE2'),
    'reject': ('AAB3BE', 'FFFFFF'),
    'tile': ('D9B8C0', 'F5E8EB'),
    'text': ('657180', 'FFFFFF'),
}

def build_figures():
    figures = {}

    def new(name, zh, en, claim):
        f = {'id': name, 'title': {'zh': zh, 'en': en}, 'claim': claim, 'nodes': [], 'edges': []}
        figures[name] = f
        return f

    def node(f, key, x, y, w, h, zh='', en='', role='neutral', font=10):
        f['nodes'].append(dict(id=key,x=x,y=y,w=w,h=h,zh=zh,en=en,role=role,font=font))

    def edge(f, a, b, ap='east', bp='west', via=(), kind='flow'):
        f['edges'].append(dict(a=a,b=b,ap=ap,bp=bp,via=list(via),kind=kind))

    def matrix(f, prefix, cx, cy, cols=5, rows=3, cell=.24, role='tile'):
        for r in range(rows):
            for c in range(cols):
                node(f,f'{prefix}{r}{c}',cx+(c-(cols-1)/2)*cell,cy+(r-(rows-1)/2)*cell,
                     cell*.85,cell*.85,role=role)

    f=new('overview','并发请求共享单卡','Concurrent requests share one GPU',
          'Concurrent requests share one GPU and receive streamed outputs; counts are illustrative.')
    node(f,'requesttitle',1.3,2.3,2.5,.65,'并发请求','Requests','text',14)
    node(f,'responsetitle',10.0,2.3,2.8,.65,'流式响应','Responses','text',14)
    node(f,'gpu',5.65,5.0,4.3,4.8,role='mechanism')
    node(f,'servicename',5.65,3.25,3.7,.7,'InferX','InferX','text',18)
    node(f,'modelname',5.65,4.05,3.9,.5,'Qwen3.8-27B','Qwen3.8-27B','text',13)
    # Resource tile: a single execution device, not a measured utilization map.
    node(f,'device',5.65,5.3,2.6,1.7,'GPU','GPU','neutral',19)
    node(f,'gpulabel',5.65,6.75,3.9,.5,'1 × 48GB RTX 4090','1 × 48GB RTX 4090','text',12)
    for i in range(3):
        y=3.65+i*1.25
        node(f,f'request{i}',1.3,y,2.0,.85,f'请求 {i+1}',f'Req. {i+1}','neutral',12)
        node(f,f'answer{i}',10.0,y,2.3,.85,role='state')
        for j in range(3):
            node(f,f'token{i}{j}',9.3+j*.5,y,.32,.3,role='state')
        node(f,f'inputport{i}',3.5,y,.01,.01,role='text')
        node(f,f'outputport{i}',7.8,y,.01,.01,role='text')
        edge(f,f'request{i}',f'inputport{i}')
        edge(f,f'outputport{i}',f'answer{i}')
    node(f,'morerequests',1.3,7.05,2.0,.6,'…','…','text',18)
    node(f,'moreanswers',10.0,7.05,2.3,.6,'…','…','text',18)
    node(f,'limit',5.65,8.15,11.2,.55,'首字与生成延迟均受SLO约束',
         'First-token and generation latency SLOs','text',13)
    node(f,'limitvalues',5.65,8.9,11.2,.5,'TTFT P95 ≤ 10s   ·   TPOT P95 ≤ 40ms',
         'TTFT P95 ≤ 10s   ·   TPOT P95 ≤ 40ms','text',12)

    f=new('problems','并发推理的两项瓶颈','Two bottlenecks in concurrent inference',
          'Shared-GPU stage contention and matrix execution cost motivate scheduling and kernel optimization. MTP concurrency is an engineering adaptation shown separately; no measured time or bottleneck shares are shown.')
    node(f,'pdtitle',2.4,2.15,4.7,.45,'I  预填充与解码干扰','I  P/D interference','text',12)
    node(f,'p',1.8,3.15,2.9,.7,'Prefill','Prefill','neutral',11)
    node(f,'d',4.0,3.15,1.0,.7,'D','D','mechanism',11)
    edge(f,'p','d')
    node(f,'wait',1.8,4.05,2.9,.55,'解码等待','Decode waits','text',11)
    node(f,'pdnote',2.4,4.9,4.7,.45,'共享GPU的执行时间','Shared GPU execution time','text',10)

    node(f,'matrixtitle',7.7,2.15,4.7,.45,'II  算子效率','II  Operator efficiency','text',12)
    matrix(f,'problemw',7.7,2.95,4,2,.24)
    node(f,'x',5.85,3.9,.5,.6,'x','x','text',12)
    node(f,'op',7.7,3.9,2.4,.7,'矩阵运算','Matrix execution','neutral',11)
    node(f,'y',9.55,3.9,.5,.6,'y','y','text',12)
    node(f,'wport',7.7,3.25,.01,.01,role='text')
    edge(f,'x','op');edge(f,'op','y');edge(f,'wport','op',ap='south',bp='north')
    node(f,'matrixnote',7.7,4.9,4.7,.45,'计算与权重访存开销','Compute and weight-access cost','text',10)

    f=new('architecture','部署与推理架构','Deployment and inference architecture',
          'One GPU executes quantized prefill/decode while per-request state feeds the host scheduler.')
    node(f,'api',.8,3.65,1.5,.5,'请求','Requests','text')
    node(f,'queue',2.55,3.65,1.2,.7)
    for i in range(3):
        node(f,f'q{i}',2.18+i*.37,3.65,.25,.5,role='neutral')
    node(f,'qlabel',2.55,4.35,1.5,.4,'等待队列','Queue','text')
    node(f,'sched',5.15,3.65,2.5,.95,'①  进度调度\n预填充 / 解码','①  Scheduler\nPrefill / decode','mechanism')
    node(f,'gpu',9.65,4.4,4.7,4.1)
    node(f,'gputitle',9.65,2.66,4.2,.4,'单张48GB RTX 4090','One 48GB RTX 4090','text')
    node(f,'exec',9.65,3.65,3.65,.95,'②  INT4计算\nMTP验证','②  INT4 kernels /\nMTP verification')
    for i,(z,e) in enumerate([('请求A  |  KV · 状态 · 进度','A  |  KV / state / progress'),
                              ('请求B  |  KV · 状态 · 进度','B  |  KV / state / progress'),
                              ('请求…  |  各自独立维护','…  |  own KV / state')]):
        node(f,f'state{i}',9.65,4.85+i*.43,3.65,.37,z,e,'state',9)
    node(f,'statecap',9.65,6.18,3.8,.35,'工程：MTP并发适配','Engineering: MTP state','text',9)
    node(f,'weights',5.15,5.55,2.5,.95,'校准后的\n4bit权重','Calibrated\n4-bit weights','mechanism')
    node(f,'out',14.0,3.65,2.0,.7,'流式响应','Streamed\nresponses','text')
    node(f,'feedbacklabel',5.4,6.7,3.1,.35,'反馈已提交进度','Committed progress','text',9)
    edge(f,'api','queue');edge(f,'queue','sched');edge(f,'sched','exec');edge(f,'exec','out')
    edge(f,'weights','exec',via=[(7.0,5.55),(7.0,4.3)],bp='south west')
    edge(f,'exec','state0',ap='south',bp='north',kind='bidirectional')
    edge(f,'state2','sched',ap='south',bp='south west',
         via=[(9.65,6.7),(3.6,6.7),(3.6,4.4)],kind='feedback')

    f=new('int4','INT4计算流程','Actual INT4 computation path',
          'Quantized weights and activations feed integer dot products; separate scales restore the output.')
    matrix(f,'wgrid',1.35,3.25)
    matrix(f,'agrid',1.35,5.3)
    node(f,'wlabel',1.35,2.42,2.3,.4,'离线权重量化','Offline weights','text')
    node(f,'alabel',1.35,6.08,2.3,.4,'在线激活量化','Online activations','text')
    node(f,'wport',2.08,3.25,.01,.01,role='text')
    node(f,'aport',2.08,5.3,.01,.01,role='text')
    node(f,'wq',4.05,3.25,2.8,.85,'H512 / GPTQ\nW4 · G512','H512 / GPTQ\nW4 · G512','mechanism')
    node(f,'aq',4.05,5.3,2.8,.85,'H512 / 动态量化\nA4 · G512','H512 / quant.\nA4 · G512','mechanism')
    node(f,'mma',7.65,4.3,2.55,2.15,'','','mechanism')
    matrix(f,'compute',7.65,4.0,4,2,.31)
    node(f,'mmalabel',7.65,4.82,2.4,.55,'INT4矩阵指令','INT4 MMA','text')
    node(f,'ep',10.9,4.3,1.75,1.05,'缩放与累加','Rescale\nAccumulate',font=9)
    matrix(f,'ogrid',13.95,4.3,4,3,.25,'neutral')
    node(f,'oport',13.35,4.3,.01,.01,role='text')
    node(f,'olabel',13.95,5.18,2.0,.4,'BF16输出','BF16 output','text')
    node(f,'ws',7.7,2.35,2.5,.35,'权重缩放因子','Weight scales','text',9)
    node(f,'as',7.7,6.25,2.5,.35,'激活缩放因子','Activation scales','text',9)
    edge(f,'wport','wq');edge(f,'aport','aq')
    edge(f,'wq','mma',bp='north west');edge(f,'aq','mma',bp='south west')
    edge(f,'mma','ep');edge(f,'ep','oport')
    edge(f,'wq','ep',ap='north',bp='north',via=[(4.05,2.35),(10.9,2.35)])
    edge(f,'aq','ep',ap='south',bp='south',via=[(4.05,6.25),(10.9,6.25)])

    f=new('state','请求状态隔离与提交','Independent request state and commit',
          'Requests accept different draft prefixes; KV and recurrent state follow their own actual commits.')
    node(f,'draftlabel',4.3,2.3,4.7,.4,'MTP候选（示意）','MTP draft tokens (illustrative)','text')
    node(f,'statelabel',11.9,2.3,4.7,.4,'独立的KV、递推状态与进度','Own KV / recurrent state / progress','text')
    for ri,(letter,y,accepted) in enumerate([('A',3.55,2),('B',5.2,1)]):
        node(f,f'req{ri}',1.25,y,1.7,.5,f'请求{letter}',f'Request {letter}','text')
        node(f,f'draftrow{ri}',4.25,y,3.35,.98)
        for j in range(3):
            node(f,f't{ri}{j}',3.1+j*1.15,y,.72,.72,f'{letter.lower()}{j+1}',f'{letter.lower()}{j+1}',
                 'accept' if j<accepted else 'reject')
        node(f,f'verify{ri}',7.4,y,1.35,.82,'验证','Verify')
        node(f,f'state{ri}',11.65,y,4.0,1.0,
             f'{letter}：接受前{accepted}个\n按提交结果更新状态',
             f'{letter}: accept {accepted} draft'+('s' if accepted>1 else '')+'\nUpdate own state',
             'state')
        edge(f,f'req{ri}',f'draftrow{ri}')
        edge(f,f'draftrow{ri}',f'verify{ri}')
        edge(f,f'verify{ri}',f'state{ri}')
    node(f,'legend',7.5,6.35,14.0,.4,'实线框：接受；虚线框：拒绝。省略回退Token，接受数与最终提交数不同。',
         'Solid: accepted; dashed: rejected. Drafts only; fallback omitted. Accepted drafts ≠ total commits.','text',9)

    f=new('schedule','分块预填充与进度调度','Chunked prefill and progress-aware scheduling',
          'Chunk boundaries permit work selection; remaining-work estimates use each request’s committed progress.')
    node(f,'inputlabel',2.2,2.8,3.1,.6,'8192 Token输入','8192-token input','text')
    node(f,'gpulabel',2.2,4.45,3.1,.8,'单卡执行顺序\n（机制示意）','One GPU\nIllustrative order','text')
    for i in range(4):
        node(f,f'chunk{i}',5.55+i*2.4,2.8,2.25,.7,'2048','2048')
        node(f,f'pf{i}',5.02+i*2.4,4.45,1.28,.8,'预填充','PF')
        node(f,f'gen{i}',6.27+i*2.4,4.45,.95,.8,'解码','Gen','mechanism')
    node(f,'budget',9.2,5.95,9.5,.72,'剩余轮次 = max（批次估计，单请求估计）',
         'Rounds = max(batch estimate, per-request estimate)','mechanism')
    node(f,'chunknote',11.25,3.6,5.5,.35,'分块之间重新调度','Select work between chunks','text',9)
    edge(f,'chunk1','pf2',ap='south',bp='north',via=[(7.95,3.6),(9.82,3.6)])
    edge(f,'budget','gen1',ap='north',bp='south')
    f=new('bounded-decode','有界解码成员选择','Bounded decode membership',
          'After the original scheduler chooses decode, eligible B10–14 states may serve the nine least-lax requests once; omitted requests require at least three full-round costs of slack, then two actual full-ready decode rounds are mandatory. Arrows show policy order, not measured timing or a pure host speedup.')
    node(f,'guard',2.1,2.5,3.6,1.4,'检查时限余量\n与就绪请求数','Check slack\nand ready count','neutral',10)
    node(f,'subset',6.65,2.5,3.7,1.4,'本轮服务\n余量最小的9个','This decode round\nNine least-lax rows','mechanism',10)
    node(f,'full1',11.15,2.5,3.6,1.4,'下一解码轮\n全部就绪请求','Next decode\nAll ready requests','state',10)
    node(f,'full2',15.55,2.5,3.6,1.4,'再下一解码轮\n全部就绪请求','Following decode\nAll ready requests','state',10)
    node(f,'eligible',4.4,1.35,3.2,.55,'触发条件满足','Eligible state','text',9)
    node(f,'fallback',2.1,4.65,3.6,1.0,'否则仍服务全员','Otherwise:\nserve all ready','neutral',10)
    node(f,'spare',6.65,4.55,4.2,1.2,'暂缓请求的余量\n≥ 3轮全员解码成本','Omitted-row slack\n≥ 3 full-round costs','text',10)
    node(f,'cadence',13.25,4.55,7.2,1.2,'必须完成两轮全员解码\n预填充不消耗这两个轮次','Complete two full decode rounds;\nprefill does not consume this count','text',10)
    edge(f,'guard','subset');edge(f,'subset','full1');edge(f,'full1','full2')
    edge(f,'guard','fallback',ap='south',bp='north')
    edge(f,'spare','subset',ap='north',bp='south')

    f=new('commit-lifecycle','逐请求裁决与提交顺序','Per-request resolution and commit order',
          'Target-verified row tokens are clipped by output policy, then pending model state is resolved before request counters and output are committed. Cancelled rows commit zero; arrows are dependencies, not timing.')
    node(f,'verify',2.0,3.0,3.1,1.3,'候选序列\n目标模型验证','Target-model\nverification','neutral',11)
    node(f,'policy',6.3,3.0,3.5,1.3,'输出裁决\nEOS / stop / 预算','Output policy\nEOS / stop / budget','mechanism',11)
    node(f,'resolve',10.8,3.0,3.5,1.3,'逐请求处理状态\n提交 / 回滚','Resolve state\nCommit / roll back','state',11)
    node(f,'publish',15.2,3.0,3.3,1.3,'更新提交记录\n发布有效输出','Update commits\nPublish output','neutral',11)
    node(f,'cancel',6.3,4.9,4.5,.7,'取消标记 → 提交长度为0','Cancellation → zero commits','text',10)
    edge(f,'verify','policy');edge(f,'policy','resolve');edge(f,'resolve','publish')
    edge(f,'cancel','policy',ap='north',bp='south')

    f=new('observability','从三类记录定位等待与异常','Correlate three observation sources',
          'Client timings, native logs and GPU samples support offline analysis of the same run; no automatic causal tracing is implied.')
    node(f,'client',3.0,2.6,5.2,1.35,'客户端时序\n首字 / 完成 / 错误',
         'Client timing\nFirst / last / errors','state',14)
    node(f,'native',3.0,4.7,5.2,1.35,'引擎请求日志\n输入 / 缓存 / 提交',
         'Engine log\nInput / cache / state','state',14)
    node(f,'gpu_samples',3.0,6.8,5.2,1.35,'GPU采样\n显存 / 利用率 / 功耗',
         'GPU samples\nMemory / load / power','state',14)
    node(f,'analysis',10.2,4.7,4.5,2.65,'同次运行\n配置与时间对照\n离线分析',
         'Compare one run\nConfig + time\nOffline analysis','mechanism',14)
    node(f,'latency',17.2,3.45,5.2,1.7,'延迟与吞吐\nP95 / 总tok/s',
         'Latency + throughput\nP95 / total tok/s','neutral',14)
    node(f,'diagnosis',17.2,5.95,5.2,1.7,'资源与异常\n对照日志排查',
         'Resources + errors\nInspect matching logs','neutral',14)
    edge(f,'client','analysis',via=[(6.8,2.6),(6.8,4.7)])
    edge(f,'native','analysis')
    edge(f,'gpu_samples','analysis',via=[(6.8,6.8),(6.8,4.7)])
    edge(f,'analysis','latency',via=[(13.5,4.7),(13.5,3.45)])
    edge(f,'analysis','diagnosis',via=[(13.5,4.7),(13.5,5.95)])
    return figures


def tex_escape(text):
    mapping={'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$'}
    return r'\\'.join(''.join(mapping.get(c,c) for c in line) for line in text.split('\n'))


def tikz_figure(f, language):
    result=[]
    def color(h):
        return '{rgb,255:red,'+str(int(h[:2],16))+';green,'+str(int(h[2:4],16))+';blue,'+str(int(h[4:],16))+'}'
    # Draw actual resource/container bounds before their child objects.
    ordered=sorted(f['nodes'],key=lambda n:0 if n['id'] in ('gpu','mma') or n['id'].startswith('draftrow') else 1)
    for n in ordered:
        role=n['role'];stroke,fill=COLORS[role]
        if role=='text':
            options=f'sm,inner sep=0pt,text width={n["w"]}cm'
        else:
            options=(f'sm,draw={color(stroke)},fill={color(fill)},'
                     f'line width=.65pt,inner sep=0pt,minimum width={n["w"]}cm,minimum height={n["h"]}cm,'
                     f'text width={max(.01,n["w"]-.12)}cm')
            if role=='reject':options+=',dashed,text=Muted'
        options+=f',font=\\fontsize{{{n["font"]}}}{{{n["font"]+2}}}\\selectfont'
        result.append(f'\\node[{options}] ({f["id"]}{n["id"]}) at ({n["x"]},{n["y"]}) {{{tex_escape(n[language])}}};')
    for e in f['edges']:
        style='feed' if e['kind']=='feedback' else 'flow'
        if e['kind']=='bidirectional':style='{Latex[length=1.1mm]}-{Latex[length=1.1mm]},draw=Blue,line width=.7pt'
        a=f'({f["id"]}{e["a"]}.{e["ap"]})';b=f'({f["id"]}{e["b"]}.{e["bp"]})'
        chain='--'.join([a]+[f'({x},{y})' for x,y in e['via']]+[b])
        result.append(f'\\draw[{style}] {chain};')
    # Redraw transparent label backgrounds over the scalar/feedback lines.
    for n in f['nodes']:
        if n['id'] in ('feedbacklabel','ws','as'):
            result.append(f'\\node[sm,fill=white,inner sep=2pt,font=\\fontsize{{9}}{{11}}\\selectfont] at ({n["x"]},{n["y"]}) {{{tex_escape(n[language])}}};')
    return '\n'.join(result)


def drawio_document(figures, language):
    doc=ET.Element('mxfile',host='app.diagrams.net',type='device')
    ports={'east':(1,.5),'west':(0,.5),'north':(.5,0),'south':(.5,1),
           'south west':(0,1),'north west':(0,0)}
    for f in figures.values():
        diagram=ET.SubElement(doc,'diagram',id=f['id'],name=f['title'][language])
        model=ET.SubElement(diagram,'mxGraphModel',dx='1450',dy='500',grid='1',gridSize='10',
                            page='1',pageScale='1',pageWidth='1450',pageHeight='500',math='0',shadow='0')
        root=ET.SubElement(model,'root')
        ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
        for n in sorted(f['nodes'],key=lambda n:0 if n['id'] in ('gpu','mma') or n['id'].startswith('draftrow') else 1):
            stroke,fill=COLORS[n['role']]
            style=(f'rounded=0;whiteSpace=wrap;html=0;align=center;verticalAlign=middle;'
                   f'fontFamily=Arial;fontSize={n["font"]*3.5278:.2f};fontColor=#20252C;'
                   f'strokeColor=#{stroke};fillColor=#{fill};strokeWidth=2;')
            if n['role']=='text':style+='strokeColor=none;fillColor=none;'
            if n['role']=='reject':style+='dashed=1;fontColor=#657180;'
            cell=ET.SubElement(root,'mxCell',id=n['id'],value=n[language],style=style,vertex='1',parent='1')
            ET.SubElement(cell,'mxGeometry',x=str((n['x']-n['w']/2-.7)*100),
                          y=str((n['y']-n['h']/2-2.0)*100),width=str(n['w']*100),height=str(n['h']*100),
                          **{'as':'geometry'})
        for i,e in enumerate(f['edges']):
            ex,ey=ports[e['ap']];tx,ty=ports[e['bp']]
            style=f'endArrow=block;html=0;rounded=0;strokeWidth=2;strokeColor=#657180;exitX={ex};exitY={ey};entryX={tx};entryY={ty};'
            if e['kind']=='feedback':style+='dashed=1;strokeColor=#32649A;'
            if e['kind']=='bidirectional':style+='startArrow=block;strokeColor=#32649A;'
            cell=ET.SubElement(root,'mxCell',id=f'edge{i}',style=style,edge='1',parent='1',source=e['a'],target=e['b'])
            geom=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
            if e['via']:
                points=ET.SubElement(geom,'Array',**{'as':'points'})
                for x,y in e['via']:ET.SubElement(points,'mxPoint',x=str((x-.7)*100),y=str((y-2.0)*100))
    return ET.tostring(doc,encoding='unicode')+'\n'
