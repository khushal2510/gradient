/* Training and fine-tuning page demos */
(function(){
  function seg(box,fn){$$("button",box).forEach(function(b){b.addEventListener("click",function(){$$("button",box).forEach(function(x){x.setAttribute("aria-pressed",x===b?"true":"false")});fn(b.dataset.k)})})}
  function sig(x){return 1/(1+Math.exp(-x))}
  function tog(btn,on,off,fn){btn.addEventListener("click",function(){var s=btn.getAttribute("aria-pressed")!=="true";btn.setAttribute("aria-pressed",s?"true":"false");btn.textContent=s?on:off;fn(s)})}
  function lab(g,t,x,y,col,al){g.fillStyle=col;g.textAlign=al||"left";g.fillText(t,x,y)}

  /* 1. pipeline */
  (function(){
    var cc=$("#ftPipe");if(!cc)return;
    var ST=[{n:"Pretrain",data:"1–20 trillion tokens of web text, code, books, maths",obj:"Predict the next token",cmp:"about 95% or more of all compute",w:.95,t:"Broad knowledge and language skill are learned here. The result is a base model: it continues text but does not reliably follow instructions."},
      {n:"Continue",data:"10–500 billion tokens of domain, long-context or multilingual text",obj:"Predict the next token",cmp:"1–5%",w:.35,t:"Optional. Adds a domain (law, medicine, a language) or longer context. Mix in general data so the model does not forget."},
      {n:"SFT",data:"10 thousand to 1 million instruction and answer pairs",obj:"Cross-entropy on the answer tokens",cmp:"well under 1%",w:.12,t:"Teaches the format: follow instructions, hold a conversation, call tools. Quality and diversity matter more than size."},
      {n:"Preference",data:"10 thousand to 1 million chosen / rejected pairs",obj:"DPO loss, or RLHF with a KL penalty",cmp:"about 1%",w:.18,t:"Pushes answers toward what people prefer: helpful, honest, safe, well formatted. A KL term keeps the model near its SFT version."},
      {n:"RL (verifiable)",data:"Prompts with checkable answers: maths, code, formats",obj:"Policy gradient (GRPO, PPO) on a pass / fail reward",cmp:"1–10% in reasoning models",w:.3,t:"The model samples long answers, gets reward when the result is correct, and learns to reason longer and check its work."}];
    var sel=0,t=0,sg=$("#pSeg");sg.innerHTML=ST.map(function(s,i){return '<button type="button" data-k="'+i+'" aria-pressed="'+(i===0)+'">'+esc(s.n)+"</button>"}).join("");
    function info(){var s=ST[sel];$("#pData").textContent=s.data;$("#pObj").textContent=s.obj;$("#pCmp").textContent=s.cmp;$("#pTxt").textContent=s.t}
    seg(sg,function(k){sel=+k;info();V.draw()});info();
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,n=ST.length,gap=w*.025,bw=(w-gap*(n+1))/n,bh=h*.34,by=h*.18,i;g.clearRect(0,0,w,h);g.font="600 "+Math.max(10,Math.min(13,w/60))+"px sans-serif";
      for(i=0;i<n;i++){var x=gap+i*(bw+gap),on=i===sel;
        g.fillStyle=on?rgba(c.c2,.16):c.surface;g.strokeStyle=on?c.c2:c.line;g.lineWidth=on?2:1;g.beginPath();g.roundRect(x,by,bw,bh,10);g.fill();g.stroke();
        lab(g,ST[i].n,x+bw/2,by+bh/2+4,on?c.ink:c.muted,"center");
        if(i<n-1){var x2=x+bw,x3=x2+gap;g.strokeStyle=c.muted;g.lineWidth=1.3;g.beginPath();g.moveTo(x2+2,by+bh/2);g.lineTo(x3-2,by+bh/2);g.stroke();
          var ph=((t/1400)+i*.23)%1;g.fillStyle=c.c1;g.beginPath();g.arc(x2+gap*ph,by+bh/2,3,0,6.3);g.fill()}
        var bm=h*.34,hh=bm*Math.pow(ST[i].w,.8);g.fillStyle=on?c.c2:rgba(c.c3,.55);g.fillRect(x+bw*.2,h*.92-hh,bw*.6,hh)}
      lab(g,"relative compute",gap,h*.97,c.muted)});
    var L=loopOn(cc,function(dt){t+=dt;V.draw()});if(!rm)L.set(true)})();

  /* 2. scaling laws */
  (function(){
    var cc=$("#ftScale");if(!cc)return;var E=1.69,A=406.4,B=410.7,al=.34,be=.28;
    function L(N,D){return E+A/Math.pow(N,al)+B/Math.pow(D,be)}
    function fmt(x){var u=["","K","M","B","T","Q"],i=0;while(x>=1000&&i<5){x/=1000;i++}return x.toPrecision(3)+u[i]}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,pad=44,C=Math.pow(10,+$("#scC").value),ln=+$("#scN").value,lo=7,hi=12.5,i,best=[1e9,0],pts=[];
      for(i=0;i<=200;i++){var l=lo+(hi-lo)*i/200,N=Math.pow(10,l),D=C/(6*N),v=L(N,D);pts.push([l,v]);if(v<best[0])best=[v,l]}
      var X=function(l){return pad+(l-lo)/(hi-lo)*(w-pad-12)},ymin=1.6,ymax=4.2,Y=function(v){return h-26-(clamp(v,ymin,ymax)-ymin)/(ymax-ymin)*(h-40)};
      g.clearRect(0,0,w,h);g.font="11px sans-serif";g.strokeStyle=c.line;g.lineWidth=1;
      for(var y=2;y<=4;y+=.5){g.beginPath();g.moveTo(pad,Y(y));g.lineTo(w-8,Y(y));g.stroke();lab(g,y.toFixed(1),pad-6,Y(y)+4,c.muted,"right")}
      for(var l2=8;l2<=12;l2++)lab(g,"10^"+l2,X(l2),h-8,c.muted,"center");lab(g,"parameters N →",w-12,12,c.muted,"right");lab(g,"loss",6,12,c.muted);
      g.strokeStyle=c.c2;g.lineWidth=2.6;g.beginPath();pts.forEach(function(p,k){k?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1]))});g.stroke();
      var mv=L(Math.pow(10,ln),C/(6*Math.pow(10,ln)));g.fillStyle=c.c1;g.beginPath();g.arc(X(ln),Y(mv),6,0,6.3);g.fill();
      g.fillStyle=c.c3;g.save();g.translate(X(best[1]),Y(best[0]));g.rotate(Math.PI/4);g.fillRect(-6,-6,12,12);g.restore();
      var Nb=Math.pow(10,best[1]),Db=C/(6*Nb);
      $("#scCV").textContent=(+$("#scC").value).toFixed(1);$("#scNV").textContent=ln.toFixed(2);
      var Nm=Math.pow(10,ln),Dm=C/(6*Nm);$("#scMine").textContent=fmt(Nm)+" params, "+fmt(Dm)+" tokens";$("#scLoss").textContent=mv.toFixed(3);
      $("#scBest").textContent=fmt(Nb)+" params, "+fmt(Db)+" tokens";$("#scBL").textContent=best[0].toFixed(3)});
    $("#scC").addEventListener("input",function(){V.draw()});$("#scN").addEventListener("input",function(){V.draw()})})();

  /* 3. loss masking */
  (function(){
    var box=$("#mkBox");if(!box)return;
    var EX=[[["<|user|>",0],["What",0],["is",0],["LoRA",0],["?",0],["<|end|>",0],["<|assistant|>",0],["Low-rank",1],["adapters",1],["that",1],["tune",1],["a",1],["frozen",1],["model",1],["cheaply",1],[".",1],["<|end|>",1]],
      [["<|user|>",0],["Hi",0],["!",0],["<|end|>",0],["<|assistant|>",0],["Hello",1],["!",1],["How",1],["can",1],["I",1],["help",1],["?",1],["<|end|>",1],["<|user|>",0],["Define",0],["epoch",0],["<|end|>",0],["<|assistant|>",0],["One",1],["full",1],["pass",1],["over",1],["the",1],["data",1],[".",1],["<|end|>",1]]];
    var mode="resp";
    function draw(){var ex=EX[+$("#mkEx").value],on=0;box.innerHTML=ex.map(function(t){var lit=mode==="all"||t[1]===1;if(lit)on++;return '<span class="tok'+(lit?" on":"")+(t[0].indexOf("<|")===0?" sp":"")+'">'+esc(t[0])+"</span>"}).join("");
      var resp=ex.filter(function(t){return t[1]}).length;$("#mkN").textContent=on;$("#mkT").textContent=ex.length;$("#mkP").textContent=Math.round(resp/on*100)+"%"}
    seg($("#mkSeg"),function(k){mode=k;draw()});$("#mkEx").addEventListener("change",draw);draw()})();

  /* 4a. LoRA rank */
  (function(){
    var cc=$("#ftLora");if(!cc)return;var n=20,U=[],Vv=[],S=[],r0=rng(11),i,j,k;
    for(k=0;k<24;k++){var u=[],v=[],a=0,b=0;for(i=0;i<n;i++){u.push(gauss(r0));v.push(gauss(r0));a+=u[i]*u[i];b+=v[i]*v[i]}a=Math.sqrt(a);b=Math.sqrt(b);U.push(u.map(function(x){return x/a}));Vv.push(v.map(function(x){return x/b}))}
    function sv(K){var s=[];for(var q=0;q<32;q++)s.push(q<K?1/Math.pow(q+1,.7):0.015/(1+q*.1));return s}
    function mat(s,R){var M=[];for(var y=0;y<n;y++){M.push([]);for(var x=0;x<n;x++){var v=0;for(var q=0;q<Math.min(R,24);q++)v+=s[q]*U[q][y]*Vv[q][x];M[y].push(v)}}return M}
    function heat(g,M,x0,y0,sz,c){var m=0,y,x;for(y=0;y<n;y++)for(x=0;x<n;x++)m=Math.max(m,Math.abs(M[y][x]));m=m||1;var cs=sz/n;for(y=0;y<n;y++)for(x=0;x<n;x++){var v=M[y][x]/m;g.fillStyle=v>0?rgba(c.c1,Math.min(1,v)):rgba(c.c2,Math.min(1,-v));g.fillRect(x0+x*cs,y0+y*cs,cs+.5,cs+.5)}g.strokeStyle=c.line;g.strokeRect(x0,y0,sz,sz)}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,r=+$("#loR").value,K=+$("#loK").value,s=sv(K),sz=Math.min(h-40,w*.27);g.clearRect(0,0,w,h);g.font="11px sans-serif";
      var full=mat(s,24),ap=mat(s,r);heat(g,full,10,26,sz,c);lab(g,"update ΔW",10,16,c.muted);heat(g,ap,sz+30,26,sz,c);lab(g,"rank-"+r+" approximation",sz+30,16,c.muted);
      var px=2*sz+60,pw=w-px-14,tot=0,q;for(q=0;q<32;q++)tot+=s[q]*s[q];var cum=0,pts=[];for(q=0;q<32;q++){cum+=s[q]*s[q];pts.push(cum/tot)}
      g.strokeStyle=c.line;g.strokeRect(px,26,pw,sz);lab(g,"energy kept vs rank",px,16,c.muted);g.strokeStyle=c.c2;g.lineWidth=2.4;g.beginPath();pts.forEach(function(v,q2){var X=px+(q2+1)/32*pw,Y=26+sz-v*sz;q2?g.lineTo(X,Y):g.moveTo(X,Y)});g.stroke();
      var X0=px+r/32*pw,Y0=26+sz-pts[r-1]*sz;g.fillStyle=c.c1;g.beginPath();g.arc(X0,Y0,5,0,6.3);g.fill();lab(g,"rank →",px+pw,h-8,c.muted,"right");
      $("#loRV").textContent=r;$("#loKV").textContent=K;$("#loP").textContent=(2*r/4096*100).toFixed(2)+"% of d² ("+(2*r*4096).toLocaleString()+" vs "+(4096*4096).toLocaleString()+")";$("#loE").textContent=(pts[r-1]*100).toFixed(1)+"%"});
    $("#loR").addEventListener("input",function(){V.draw()});$("#loK").addEventListener("input",function(){V.draw()})})();

  /* 4b. memory */
  (function(){
    var cc=$("#ftMem");if(!cc)return;var P={1:[1e9,2048,16],3:[3e9,3072,28],7:[7e9,4096,32],13:[13e9,5120,40],34:[34e9,8192,48],70:[70e9,8192,80]},ck=true;
    function calc(){var m=P[$("#mmM").value],N=m[0],d=m[1],L=m[2],r=+$("#mmR").value,tok=+$("#mmK").value,t=$("#mmT").value,GB=1e9,Pl=L*19*d*r,w,gr,op,tr;
      if(t==="full"){w=2*N;gr=2*N;op=12*N;tr=N}else{w=(t==="lora"?2*N:0.56*N)+4*Pl;gr=4*Pl;op=8*Pl;tr=Pl}
      var act=ck?(2*d*L*tok+34*d*tok):34*d*L*tok;return {w:w/GB,g:gr/GB,o:op/GB,a:act/GB,tr:tr,r:r,tok:tok}}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,x=calc(),tot=x.w+x.g+x.o+x.a,mx=Math.max(100,tot*1.08),pad=10,W=w-2*pad,X=function(v){return pad+v/mx*W},segs=[["weights",x.w,c.c1],["gradients",x.g,c.c3],["optimizer",x.o,c.c2],["activations",x.a,c.c4]],cur=0;
      g.clearRect(0,0,w,h);g.font="11px sans-serif";var by=h*.3,bh=h*.28;
      segs.forEach(function(s,i){g.fillStyle=s[2];g.fillRect(X(cur),by,Math.max(0,X(cur+s[1])-X(cur)-1),bh);cur+=s[1];g.fillStyle=s[2];g.fillRect(pad+i*110,h*.82,10,10);lab(g,s[0]+" "+s[1].toFixed(1)+" GB",pad+i*110+14,h*.82+9,c.muted)});
      [[24,"24 GB"],[40,"40 GB"],[80,"80 GB"]].forEach(function(l){g.strokeStyle=c.muted;g.setLineDash([4,4]);g.beginPath();g.moveTo(X(l[0]),by-14);g.lineTo(X(l[0]),by+bh+8);g.stroke();g.setLineDash([]);lab(g,l[1],X(l[0]),by-18,c.muted,"center")});
      $("#mmRV").textContent=x.r;$("#mmKV").textContent=x.tok;$("#mmTot").textContent=tot.toFixed(1)+" GB";$("#mmTr").textContent=x.tr>=1e9?(x.tr/1e9).toFixed(2)+"B":(x.tr/1e6).toFixed(0)+"M";
      var f=[[16,"16 GB"],[24,"24 GB"],[40,"40 GB"],[48,"48 GB"],[80,"80 GB"],[160,"2×80 GB"],[640,"8×80 GB"]].find(function(q){return tot*1.1<=q[0]});$("#mmFit").textContent=f?f[1]:"more than 8×80 GB"});
    ["mmM","mmT"].forEach(function(id){$("#"+id).addEventListener("change",function(){V.draw()})});["mmR","mmK"].forEach(function(id){$("#"+id).addEventListener("input",function(){V.draw()})});
    tog($("#mmCk"),"Gradient checkpointing on","Gradient checkpointing off",function(s){ck=s;V.draw()})})();

  /* 5. 3D landscape */
  (function(){
    var cc=$("#ftLand");if(!cc)return;var a=[-1.4,.9],b=[1.3,-.7],u=[b[0]-a[0],b[1]-a[1]],ul=Math.hypot(u[0],u[1]),N=18,FM=14,R=[],acc=0;u=[u[0]/ul,u[1]/ul];
    function LB(x,y){return .5*((x-b[0])*(x-b[0])+3*(y-b[1])*(y-b[1]))}function LA(x,y){return .5*((x-a[0])*(x-a[0])+(y-a[1])*(y-a[1]))}
    function reset(){R=[0,1,2].map(function(){return {p:a.slice(),t:0,tr:[a.slice()]}})}
    function step(){var lr=+$("#ldL").value;R.forEach(function(q,k){var gx=q.p[0]-b[0],gy=3*(q.p[1]-b[1]);
      if(k===0){q.p[0]-=lr*gx;q.p[1]-=lr*gy}else if(k===1){var l2=6*lr;q.p[0]-=l2*gx;q.p[1]-=l2*gy}else{q.t-=lr*(gx*u[0]+gy*u[1]);q.p=[a[0]+q.t*u[0],a[1]+q.t*u[1]]}
      q.p[0]=clamp(q.p[0],-2.4,2.4);q.p[1]=clamp(q.p[1],-2.4,2.4);q.tr.push(q.p.slice());if(q.tr.length>200)q.tr.shift()})}
    var W=function(x,y){return [x*.55,Math.min(LB(x,y),FM)*.045-.5,y*.55]};
    var S=G3(cc,function(S,c){var A=rgbOf(c.c2),B=rgbOf(c.c1),i,j;
      for(i=0;i<N;i++)for(j=0;j<N;j++){var x0=-2.5+5*i/N,x1=-2.5+5*(i+1)/N,y0=-2.5+5*j/N,y1=-2.5+5*(j+1)/N,m=LB((x0+x1)/2,(y0+y1)/2),t=clamp(m/FM*1.4,0,1);
        S.quad([W(x0,y0),W(x1,y0),W(x1,y1),W(x0,y1)],"rgb("+Math.round(18+A[0]*t*.55+B[0]*t*t*.3)+","+Math.round(8+A[1]*t*.4+B[1]*t*t*.45)+","+Math.round(14+A[2]*t*.4)+")","rgba(255,255,255,.06)",.9)}
      var prev=null;for(i=0;i<=36;i++){var an=i/36*6.2832,px=a[0]+.55*Math.cos(an),py=a[1]+.55*Math.sin(an),p=W(px,py);p[1]+=.14;if(prev)S.line(prev,p,c.ink,2,.9);prev=p}
      var cols=[c.c1,c.c2,c.c3];R.forEach(function(q,k){for(var n=1;n<q.tr.length;n++){var p1=W(q.tr[n-1][0],q.tr[n-1][1]),p2=W(q.tr[n][0],q.tr[n][1]);p1[1]+=.16;p2[1]+=.16;S.line(p1,p2,cols[k],2.2,.95)}var p=W(q.p[0],q.p[1]);p[1]+=.2;S.dot(p,4.6,cols[k],1)});
      var pa=W(a[0],a[1]);pa[1]+=.45;S.text(pa,"pretrained",c.ink,11);var pb=W(b[0],b[1]);pb[1]+=.4;S.text(pb,"new task optimum",c.c1,11)},{yaw:.5,pitch:.95,dist:5,zoom:1.15,spin:.00012,tick:function(dt){acc+=dt;while(acc>=100){step();acc-=100}
        R.forEach(function(q,k){$("#ld"+k).textContent=LB(q.p[0],q.p[1]).toFixed(2)+" / "+LA(q.p[0],q.p[1]).toFixed(2)})}});
    $("#ldL").addEventListener("input",function(){$("#ldLV").textContent=(+this.value).toFixed(2)});$("#ldRe").addEventListener("click",function(){reset();S.draw()});reset()})();

  /* 6. DPO */
  (function(){
    var cc=$("#ftDpo");if(!cc)return;
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,pad=40,be=+$("#dpB").value,D=+$("#dpD").value,X=function(d){return pad+(d+60)/120*(w-pad-12)},Y=function(v){return h-24-clamp(v,0,6)/6*(h-40)},Y2=function(v){return h-24-v*(h-40)},i;
      g.clearRect(0,0,w,h);g.font="11px sans-serif";g.strokeStyle=c.line;for(i=0;i<=6;i+=2){g.beginPath();g.moveTo(pad,Y(i));g.lineTo(w-8,Y(i));g.stroke();lab(g,i,pad-6,Y(i)+4,c.muted,"right")}
      g.strokeStyle=c.muted;g.beginPath();g.moveTo(X(0),10);g.lineTo(X(0),h-24);g.stroke();lab(g,"Δ →",w-12,h-8,c.muted,"right");lab(g,"loss",4,12,c.muted);
      g.lineWidth=2.6;g.strokeStyle=c.c2;g.beginPath();for(i=-60;i<=60;i++){var l=Math.log(1+Math.exp(-be*i));i===-60?g.moveTo(X(i),Y(l)):g.lineTo(X(i),Y(l))}g.stroke();
      g.lineWidth=2;g.strokeStyle=c.c1;g.setLineDash([5,4]);g.beginPath();for(i=-60;i<=60;i++){var gw=sig(-be*i);i===-60?g.moveTo(X(i),Y2(gw)):g.lineTo(X(i),Y2(gw))}g.stroke();g.setLineDash([]);
      lab(g,"loss",X(-58),Y(Math.log(1+Math.exp(be*58)))+14,c.c2);lab(g,"gradient weight σ(−βΔ)",X(8),Y2(.97)+2,c.c1);
      var L0=Math.log(1+Math.exp(-be*D));g.fillStyle=c.ink;g.beginPath();g.arc(X(D),Y(L0),6,0,6.3);g.fill();
      $("#dpBV").textContent=be.toFixed(2);$("#dpDV").textContent=D;$("#dpL").textContent=L0.toFixed(3);$("#dpP").textContent=(sig(be*D)*100).toFixed(1)+"%";$("#dpG").textContent=sig(-be*D).toFixed(3)});
    $("#dpB").addEventListener("input",function(){V.draw()});$("#dpD").addEventListener("input",function(){V.draw()})})();

  /* 7a. best-of-n */
  (function(){
    var cc=$("#ftBon");if(!cc)return;var M=500,NM=256,r=rng(5),Q=[],H=[],i,j;
    for(i=0;i<M;i++){var qq=new Float32Array(NM),hh=new Float32Array(NM);for(j=0;j<NM;j++){qq[j]=gauss(r);hh[j]=-Math.log(1-r()+1e-12)}Q.push(qq);H.push(hh)}
    var cache=null,ck=-1;
    function curves(e){if(ck===e)return cache;var P=new Float64Array(NM),T=new Float64Array(NM);
      for(i=0;i<M;i++){var bp=-1e9,bt=0;for(j=0;j<NM;j++){var h=H[i][j]*e,pr=Q[i][j]+h;if(pr>bp){bp=pr;bt=Q[i][j]-.9*h}P[j]+=bp;T[j]+=bt}}
      for(j=0;j<NM;j++){P[j]/=M;T[j]/=M}ck=e;cache={P:P,T:T};return cache}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,pad=40,e=+$("#bnE").value,x=+$("#bnN").value,K=curves(e),n=Math.round(Math.pow(2,x)),lo=-1.5,hi=3.5,
        X=function(l){return pad+l/8*(w-pad-12)},Y=function(v){return h-24-(clamp(v,lo,hi)-lo)/(hi-lo)*(h-40)};
      g.clearRect(0,0,w,h);g.font="11px sans-serif";g.strokeStyle=c.line;for(i=-1;i<=3;i++){g.beginPath();g.moveTo(pad,Y(i));g.lineTo(w-8,Y(i));g.stroke();lab(g,i,pad-6,Y(i)+4,c.muted,"right")}
      for(i=0;i<=8;i+=2)lab(g,Math.pow(2,i),X(i),h-8,c.muted,"center");lab(g,"samples n (log scale) →",w-12,12,c.muted,"right");
      function line(arr,col){g.strokeStyle=col;g.lineWidth=2.6;g.beginPath();for(var k=0;k<=16;k++){var l=k/2,v=arr[Math.round(Math.pow(2,l))-1];k?g.lineTo(X(l),Y(v)):g.moveTo(X(l),Y(v))}g.stroke()}
      line(K.P,c.c1);line(K.T,c.c2);lab(g,"proxy reward",X(.2),Y(K.P[2])-10,c.c1);lab(g,"true quality",X(5.4),Y(K.T[40])+16,c.c2);
      g.strokeStyle=c.muted;g.setLineDash([4,4]);g.beginPath();g.moveTo(X(x),10);g.lineTo(X(x),h-24);g.stroke();g.setLineDash([]);
      $("#bnNV").textContent=n;$("#bnEV").textContent=e.toFixed(2);$("#bnR1").textContent=n;$("#bnR2").textContent=K.P[n-1].toFixed(2);$("#bnR3").textContent=K.T[n-1].toFixed(2);$("#bnR4").textContent=(Math.log(n)-(n-1)/n).toFixed(2)});
    $("#bnN").addEventListener("input",function(){V.draw()});$("#bnE").addEventListener("input",function(){V.draw()})})();

  /* 7b. GRPO */
  (function(){
    var cc=$("#ftGrpo");if(!cc)return;var U=[];function fresh(){U=[];for(var i=0;i<16;i++)U.push(Math.random())}fresh();
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,p=+$("#grP").value,G=+$("#grG").value,R=U.slice(0,G).map(function(x){return x<p?1:0}),m=R.reduce(function(a,b){return a+b},0)/G,
        sd=Math.sqrt(R.reduce(function(a,b){return a+(b-m)*(b-m)},0)/G),A=R.map(function(x){return sd>0?(x-m)/(sd+1e-6):0}),bw=(w-30)/G,mid=h*.66,i;
      g.clearRect(0,0,w,h);g.font="11px sans-serif";lab(g,"reward (0 or 1)",8,12,c.muted);lab(g,"advantage",8,mid+14,c.muted);g.strokeStyle=c.line;g.beginPath();g.moveTo(8,mid);g.lineTo(w-8,mid);g.stroke();
      for(i=0;i<G;i++){var x=18+i*bw;g.fillStyle=R[i]?c.c1:rgba(c.c1,.15);g.fillRect(x+3,mid-8-R[i]*(h*.4)-4,bw-6,Math.max(4,R[i]*(h*.4)));
        var a=A[i],ah=Math.abs(a)*(h*.12);g.fillStyle=a>=0?c.c3:c.c2;g.fillRect(x+3,a>=0?mid+4:mid+4,bw-6,Math.max(2,ah)*(a===0?0:1))}
      $("#grPV").textContent=p.toFixed(2);$("#grGV").textContent=G;$("#grM").textContent=m.toFixed(2);$("#grS").textContent=sd.toFixed(2);$("#grU").textContent=sd>0?"Yes: "+R.filter(Boolean).length+" right, "+R.filter(function(x){return !x}).length+" wrong":"No: all answers got the same reward"});
    ["grP","grG"].forEach(function(id){$("#"+id).addEventListener("input",function(){V.draw()})});$("#grGo").addEventListener("click",function(){fresh();V.draw()})})();

  /* 8. quantisation */
  (function(){
    var cc=$("#ftQuant");if(!cc)return;var n=64,W=[],mode="t",out=true;
    function gen(){var r=rng(Math.floor(Math.random()*9999)+1);W=[];for(var i=0;i<n;i++)W.push(gauss(r)*.5);W[9]=3.2}
    gen();
    function qz(bits,grp){var Q=[],L=Math.pow(2,bits-1)-1,gs=grp?16:n,i,j;for(i=0;i<n;i+=gs){var m=0,ws=W.slice(i,i+gs).map(function(v,k){return out||(i+k!==9)?v:W[i+k]});for(j=0;j<gs;j++)m=Math.max(m,Math.abs(vv(i+j)));var s=m/L||1;for(j=0;j<gs;j++)Q.push(Math.round(vv(i+j)/s)*s)}return Q}
    function vv(i){return (i===9&&!out)?.2:W[i]}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,bits=+$("#qzB").value,Q=qz(bits,mode==="g"),i,mse=0,mid=h/2,sc=(h/2-14)/3.6,bw=(w-20)/n;g.clearRect(0,0,w,h);g.strokeStyle=c.line;g.beginPath();g.moveTo(8,mid);g.lineTo(w-8,mid);g.stroke();
      for(i=0;i<n;i++){var x=10+i*bw+bw/2,v=vv(i);mse+=(Q[i]-v)*(Q[i]-v);g.strokeStyle=c.c2;g.lineWidth=Math.max(2,bw*.5);g.beginPath();g.moveTo(x,mid);g.lineTo(x,mid-Q[i]*sc);g.stroke();g.fillStyle=c.c1;g.beginPath();g.arc(x,mid-v*sc,Math.max(2,bw*.2),0,6.3);g.fill()}
      mse/=n;$("#qzBV").textContent=bits;$("#qzE").textContent=mse.toExponential(2);$("#qzM").textContent=Math.round((bits+(mode==="g"?1:0))/16*100)+"%"});
    $("#qzB").addEventListener("input",function(){V.draw()});seg($("#qzSeg"),function(k){mode=k;V.draw()});tog($("#qzOut"),"Outlier on","Outlier off",function(s){out=s;V.draw()});$("#qzNew").addEventListener("click",function(){gen();V.draw()})})();

  /* 9. chooser */
  (function(){
    var G=$("#gdG");if(!G)return;
    function go(){var g=G.value,d=$("#gdD").value,h=$("#gdH").value,L=[];
      var meth=h==="small"?"QLoRA (4-bit base, LoRA adapters)":h==="mid"?"LoRA, or full fine-tuning for models up to about 3B":"Full fine-tuning is possible, LoRA is still a strong default";
      if(g==="style"){L.push("Use SFT with "+meth+".");L.push(d==="few"?"A few hundred high-quality examples is often enough for style and format. Keep them consistent.":"Filter for quality and diversity before adding more data.");L.push("Try prompting with examples first. If that already works, you may not need training.")}
      else if(g==="know"){L.push("Fine-tuning is a poor way to add facts. Start with retrieval (RAG) so the model reads your documents at answer time.");L.push(d==="big"?"With a large corpus, continued pretraining on it, followed by a short SFT pass, can work. Mix in general data.":"Use RAG now. Add LoRA only to teach the model how to use the retrieved text.")}
      else if(g==="task"){L.push("Use SFT with "+meth+" on examples of the exact task.");L.push(d==="few"?"With under 1,000 examples, use LoRA with a low rank, few epochs and a held-out set.":"Hold out a test set, and compare against a prompted baseline.");L.push("Check that general ability has not dropped.")}
      else if(g==="pref"){L.push("Run SFT first, then DPO on chosen / rejected pairs.");L.push(d==="few"?"Under 1,000 pairs is thin. Use a small learning rate, and consider generating pairs from your own model and ranking them.":"Use a very small learning rate, and track the chosen log-probability.");L.push("RLHF with PPO is only worth its complexity if you need online optimisation against a reward model.")}
      else if(g==="reason"){L.push("Use RL with verifiable rewards (GRPO) on prompts that have checkable answers.");L.push(h==="small"?"On one GPU, use small models (1–3B) with LoRA, short completions and few samples per prompt.":"Sample 8 or more completions per prompt and filter prompts that are always right or always wrong.");L.push("Start from a model that already gets some answers right, or the reward is always zero.")}
      else{L.push("Quantise to 8 or 4 bits (AWQ or GPTQ) first. It is the cheapest option.");L.push("If you need a much smaller model, distil from a larger teacher on your own task data.");L.push("Check quality on your own evaluation set at each step.")}
      $("#gdOut").innerHTML="<h4>Suggested approach</h4><ul>"+L.map(function(x){return "<li>"+esc(x)+"</li>"}).join("")+"</ul>"}
    ["gdG","gdD","gdH"].forEach(function(id){$("#"+id).addEventListener("change",go)});go()})();
})();
