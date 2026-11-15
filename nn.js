/* Neural network demos */
(function(){
  function f(n,d){return isFinite(n)?n.toFixed(d):"∞"}
  var sigm=function(x){return 1/(1+Math.exp(-x))};

  /* activation explorer */
  (function(){
    var cc=$("#actCv");if(!cc)return;
    var F={relu:function(x){return Math.max(0,x)},leaky:function(x){return x>0?x:0.1*x},sigmoid:sigm,tanh:Math.tanh,
      gelu:function(x){return 0.5*x*(1+Math.tanh(0.79788456*(x+0.044715*x*x*x)))},silu:function(x){return x*sigm(x)},softplus:function(x){return Math.log(1+Math.exp(x))}};
    var N={relu:"Cheap and effective. Zero for negative inputs, so those neurons pass no gradient.",leaky:"Keeps a small slope (0.1 here) for negative inputs so neurons cannot fully die.",sigmoid:"Output between 0 and 1. Slope peaks at 0.25 and vanishes in the tails. Good for a final binary probability.",tanh:"Zero-centred version of sigmoid with output between −1 and 1. Still saturates.",gelu:"Smooth ReLU-like curve used in BERT and GPT-style transformers.",silu:"x times sigmoid(x), also called Swish. Smooth and used in many modern networks.",softplus:"A smooth approximation of ReLU. Always positive, and its derivative is the sigmoid."};
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,fn=F[$("#actSel").value],X=function(x){return (x+5)/10*w},Y=function(y){return h-(y+2)/7*h};g.clearRect(0,0,w,h);
      g.strokeStyle=c.line;g.lineWidth=1;g.fillStyle=c.muted;g.font="11px sans-serif";
      for(var x=-5;x<=5;x++){g.beginPath();g.moveTo(X(x),0);g.lineTo(X(x),h);g.globalAlpha=x===0?1:.4;g.stroke();g.globalAlpha=1;if(x%1===0&&x!==0)g.fillText(x,X(x)+3,Y(0)+12)}
      for(var y=-2;y<=5;y++){g.beginPath();g.moveTo(0,Y(y));g.lineTo(w,Y(y));g.globalAlpha=y===0?1:.4;g.stroke();g.globalAlpha=1}
      function plot(fun,col){g.strokeStyle=col;g.lineWidth=2.6;g.beginPath();for(var i=0;i<=200;i++){var xx=-5+10*i/200,yy=fun(xx);i?g.lineTo(X(xx),Y(yy)):g.moveTo(X(xx),Y(yy))}g.stroke()}
      var d=function(x){return (fn(x+1e-4)-fn(x-1e-4))/2e-4};plot(fn,c.c1);plot(d,c.c2);
      var xv=+$("#actX").value;g.strokeStyle=c.ink;g.setLineDash([4,4]);g.lineWidth=1;g.beginPath();g.moveTo(X(xv),0);g.lineTo(X(xv),h);g.stroke();g.setLineDash([]);dot(g,X(xv),Y(fn(xv)),6,c.c1,c.bg);dot(g,X(xv),Y(d(xv)),6,c.c2,c.bg)});
    function upd(){var fn=F[$("#actSel").value],x=+$("#actX").value;$("#actXV").textContent=x.toFixed(2);$("#actF").textContent=fn(x).toFixed(4);$("#actD").textContent=((fn(x+1e-4)-fn(x-1e-4))/2e-4).toFixed(4);$("#actNote").textContent=N[$("#actSel").value];V.draw()}
    $("#actSel").addEventListener("change",upd);$("#actX").addEventListener("input",upd);upd()})();

  /* softmax */
  (function(){
    var box=$("#smBars");if(!box)return;var ids=["smA","smB","smC","smD"],names=["A","B","C","D"];
    box.innerHTML=names.map(function(n,i){return '<div class="bar-r" id="smr'+i+'"><span>Class '+n+'</span><div class="t"><i></i></div><span class="pc">0%</span></div>'}).join("");
    function upd(){var T=+$("#smT").value,y=+$("#smY").value,z=ids.map(function(id,i){var v=+$("#"+id).value;$("#"+id+"V").textContent=v.toFixed(1);return v});$("#smTV").textContent=T.toFixed(1);
      var m=Math.max.apply(null,z),e=z.map(function(v){return Math.exp((v-m)/T)}),s=e.reduce(function(a,b){return a+b},0),p=e.map(function(v){return v/s});
      p.forEach(function(v,i){var r=$("#smr"+i);$("i",r).style.width=(v*100)+"%";$("i",r).style.background=i===y?"var(--c2)":"var(--c1)";$(".pc",r).textContent=(v*100).toFixed(1)+"%"});
      $("#smL").textContent=(-Math.log(Math.max(p[y],1e-12))).toFixed(3);var H=0;p.forEach(function(v){if(v>0)H-=v*Math.log(v)});$("#smE").textContent=H.toFixed(3)+" nats"}
    ids.concat(["smT","smY"]).forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* backprop */
  (function(){
    var tb=$("#bpTbl");if(!tb)return;var ids=["bpX1","bpX2","bpW1","bpW2","bpB","bpY"];
    function val(){var o={};ids.forEach(function(id){o[id]=+$("#"+id).value;$("#"+id+"V").textContent=Number.isInteger(o[id])?o[id]:o[id].toFixed(1)});return o}
    function fwd(v,w1){var z=(w1==null?v.bpW1:w1)*v.bpX1+v.bpW2*v.bpX2+v.bpB,a=sigm(z),L=(a-v.bpY)*(a-v.bpY);return {z:z,a:a,L:L}}
    function grads(v){var r=fwd(v),dLa=2*(r.a-v.bpY),daz=r.a*(1-r.a),dLz=dLa*daz;return {r:r,dLa:dLa,daz:daz,dLz:dLz,w1:dLz*v.bpX1,w2:dLz*v.bpX2,b:dLz}}
    function upd(){var v=val(),G=grads(v),r=G.r,n=function(x){return x.toFixed(4)};
      tb.innerHTML=[["z = w₁x₁ + w₂x₂ + b",n(r.z),n(G.dLz)+"  (= ∂L/∂a × a(1−a))"],["a = σ(z)",n(r.a),n(G.dLa)+"  (= 2(a − y))"],["L = (a − y)²",n(r.L),"1"],["w₁",v.bpW1.toFixed(2),n(G.w1)+"  (= ∂L/∂z × x₁)"],["w₂",v.bpW2.toFixed(2),n(G.w2)+"  (= ∂L/∂z × x₂)"],["b",v.bpB.toFixed(2),n(G.b)+"  (= ∂L/∂z)"]].map(function(r){return "<tr><td><code>"+esc(r[0])+"</code></td><td>"+r[1]+"</td><td>"+r[2]+"</td></tr>"}).join("");
      var h=1e-5,num=(fwd(v,v.bpW1+h).L-fwd(v,v.bpW1-h).L)/(2*h);$("#bpChk").textContent="Numerical check for w₁: finite difference "+num.toFixed(6)+" versus analytic "+G.w1.toFixed(6)+"."}
    ids.forEach(function(id){$("#"+id).addEventListener("input",upd)});
    $("#bpStep").addEventListener("click",function(){var v=val(),G=grads(v);[["bpW1",G.w1],["bpW2",G.w2],["bpB",G.b]].forEach(function(p){var e=$("#"+p[0]);e.value=clamp(+e.value-0.5*p[1],-2,2).toFixed(2)});upd()});
    upd()})();

  /* playground */
  (function(){
    var cc=$("#pgCv");if(!cc)return;
    var DATA,net,run=false,ep=0,hist=[],acc=0;
    function spiral(n,noise,seed){var r=rng(seed),P=[],i;for(i=0;i<n;i++){var cls=i%2,t=0.12+0.88*r(),a=4.2*t+cls*Math.PI,rad=0.06+0.4*t;P.push({x:0.5+Math.cos(a)*rad+gauss(r)*noise*0.5,y:0.5+Math.sin(a)*rad+gauss(r)*noise*0.5,c:cls})}return P}
    function gen(){var k=$("#pgD").value,n=+$("#pgN").value;DATA=k==="spiral"?spiral(220,n,5):blobs(k,200,n,5)}
    var act={tanh:[Math.tanh,function(a,z){return 1-a*a}],relu:[function(z){return z>0?z:0},function(a,z){return z>0?1:0}]};
    function Net(sizes,type){this.s=sizes;this.L=sizes.length-1;this.type=type;this.W=[];this.b=[];this.m=[];this.v=[];this.t=0;
      var rr=rng(Math.floor(Math.random()*1e6)),l,k;for(l=0;l<this.L;l++){var ni=sizes[l],no=sizes[l+1],sd=Math.sqrt((type==="relu"&&l<this.L-1?2:1)/ni),W=new Float64Array(ni*no);for(k=0;k<W.length;k++)W[k]=gauss(rr)*sd;this.W.push(W);this.b.push(new Float64Array(no));this.m.push([new Float64Array(W.length),new Float64Array(no)]);this.v.push([new Float64Array(W.length),new Float64Array(no)])}}
    Net.prototype.fwd=function(x,cache){var a=x,A=[x],Z=[],l,o,i,F=act[this.type][0];for(l=0;l<this.L;l++){var ni=this.s[l],no=this.s[l+1],z=new Float64Array(no);for(o=0;o<no;o++){var s=this.b[l][o];for(i=0;i<ni;i++)s+=this.W[l][o*ni+i]*a[i];z[o]=s}Z.push(z);a=l===this.L-1?z:z.map(F);A.push(a)}if(cache){cache.A=A;cache.Z=Z}return a[0]};
    Net.prototype.step=function(batch,lr){var gW=this.W.map(function(w){return new Float64Array(w.length)}),gb=this.b.map(function(b){return new Float64Array(b.length)}),self=this,D=act[this.type][1];
      batch.forEach(function(p){var ca={},out=self.fwd([(p.x-.5)*2.5,(p.y-.5)*2.5],ca),d=[sigm(out)-p.c],l,o,i;
        for(l=self.L-1;l>=0;l--){var ni=self.s[l],no=self.s[l+1],A=ca.A[l],nd=new Array(ni).fill(0);
          for(o=0;o<no;o++){gb[l][o]+=d[o];for(i=0;i<ni;i++){gW[l][o*ni+i]+=d[o]*A[i];nd[i]+=self.W[l][o*ni+i]*d[o]}}
          if(l>0){for(i=0;i<ni;i++)nd[i]*=D(A[i],ca.Z[l-1][i])}d=nd}});
      this.t++;var n=batch.length,b1=0.9,b2=0.999,c1=1-Math.pow(b1,this.t),c2=1-Math.pow(b2,this.t);
      for(var l2=0;l2<this.L;l2++)[[this.W[l2],gW[l2],this.m[l2][0],this.v[l2][0]],[this.b[l2],gb[l2],this.m[l2][1],this.v[l2][1]]].forEach(function(q){var P=q[0],G=q[1],M=q[2],V=q[3];for(var k=0;k<P.length;k++){var g=G[k]/n;M[k]=b1*M[k]+(1-b1)*g;V[k]=b2*V[k]+(1-b2)*g*g;P[k]-=lr*(M[k]/c1)/(Math.sqrt(V[k]/c2)+1e-8)}})};
    function build(){var h1=+$("#pgH1").value,h2=+$("#pgH2").value,s=[2];if(h1>0){s.push(h1);if(h2>0)s.push(h2)}s.push(1);net=new Net(s,$("#pgAct").value);ep=0;hist=[];measure()}
    function measure(){var L=0,ok=0;DATA.forEach(function(p){var o=net.fwd([(p.x-.5)*2.5,(p.y-.5)*2.5]),q=sigm(o);L-=p.c?Math.log(Math.max(q,1e-9)):Math.log(Math.max(1-q,1e-9));if((q>.5?1:0)===p.c)ok++});L/=DATA.length;acc=ok/DATA.length;hist.push(L);if(hist.length>400)hist.shift();return L}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!net)return;heatmap(g,w,h,function(x,y){return sigm(net.fwd([(x-.5)*2.5,(y-.5)*2.5]))},c.c1,c.c2,9,.3);DATA.forEach(function(p){dot(g,p.x*w,(1-p.y)*h,4.5,p.c?c.c2:c.c1,c.bg)})});
    var LV=cv($("#pgLoss"),function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);g.strokeStyle=c.line;g.lineWidth=1;for(var k=0;k<=3;k++){var y=h-8-k/3*(h-18);g.beginPath();g.moveTo(30,y);g.lineTo(w,y);g.stroke();g.fillStyle=c.muted;g.font="10px sans-serif";g.fillText((k*0.25).toFixed(2),0,y+3)}
      g.strokeStyle=c.c2;g.lineWidth=2.2;g.beginPath();hist.forEach(function(v,i){var x=30+i/Math.max(hist.length-1,1)*(w-34),y2=h-8-clamp(v/0.75,0,1)*(h-18);i?g.lineTo(x,y2):g.moveTo(x,y2)});g.stroke()});
    function show(){V.draw();LV.draw();$("#pgE").textContent=ep;$("#pgL").textContent=hist.length?hist[hist.length-1].toFixed(3):"-";$("#pgA").textContent=(acc*100).toFixed(0)+"%"}
    var Lp=loopOn(cc,function(){if(!run)return;var lr=+$("#pgLr").value;for(var e=0;e<2;e++){var idx=DATA.map(function(_,i){return i});for(var i=idx.length-1;i>0;i--){var j=Math.floor(Math.random()*(i+1)),t=idx[i];idx[i]=idx[j];idx[j]=t}
        for(var k=0;k<idx.length;k+=20)net.step(idx.slice(k,k+20).map(function(q){return DATA[q]}),lr);ep++}measure();show()});
    $("#pgRun").addEventListener("click",function(){run=!run;this.textContent=run?"Pause":"Train";Lp.set(run)});
    $("#pgReset").addEventListener("click",function(){build();show()});
    ["pgH1","pgH2","pgAct"].forEach(function(id){$("#"+id).addEventListener("input",function(){$("#pgH1V").textContent=$("#pgH1").value;$("#pgH2V").textContent=$("#pgH2").value;build();show()})});
    ["pgD","pgN"].forEach(function(id){$("#"+id).addEventListener("input",function(){$("#pgNV").textContent=(+$("#pgN").value).toFixed(2);gen();build();show()})});
    $("#pgLr").addEventListener("input",function(){$("#pgLrV").textContent=(+this.value).toFixed(3)});
    gen();build();show();run=!rm;if(run){$("#pgRun").textContent="Pause";Lp.set(true)}})();

  /* 3D optimiser race */
  (function(){
    var cc=$("#opCv");if(!cc)return;
    var fx=function(x,y){return 0.25*x*x+1.5*y*y+0.25*Math.sin(2.5*x)*Math.sin(2.5*y)},
        gx=function(x,y){return 0.5*x+0.625*Math.cos(2.5*x)*Math.sin(2.5*y)},gy=function(x,y){return 3*y+0.625*Math.sin(2.5*x)*Math.cos(2.5*y)};
    var start=[-2.2,1.6],O=[],run=true,acc=0,N=20,FM=11.5;
    function reset(s){start=s||start;O=[0,1,2,3].map(function(k){return {p:start.slice(),v:[0,0],s:[0,0],m:[0,0],t:0,trail:[start.slice()]}});}
    function step(){var lr=+$("#opLr").value;O.forEach(function(q,k){var g=[gx(q.p[0],q.p[1]),gy(q.p[0],q.p[1])],d;q.t++;
      for(d=0;d<2;d++){if(k===0)q.p[d]-=lr*g[d];
        else if(k===1){q.v[d]=0.9*q.v[d]+g[d];q.p[d]-=lr*0.4*q.v[d]}
        else if(k===2){q.s[d]=0.9*q.s[d]+0.1*g[d]*g[d];q.p[d]-=lr*0.5*g[d]/(Math.sqrt(q.s[d])+1e-8)}
        else{q.m[d]=0.9*q.m[d]+0.1*g[d];q.s[d]=0.999*q.s[d]+0.001*g[d]*g[d];q.p[d]-=lr*0.5*(q.m[d]/(1-Math.pow(0.9,q.t)))/(Math.sqrt(q.s[d]/(1-Math.pow(0.999,q.t)))+1e-8)}
        q.p[d]=clamp(q.p[d],-2.5,2.5)}
      q.trail.push(q.p.slice());if(q.trail.length>260)q.trail.shift()})}
    var W=function(x,y){return [x*0.55,fx(x,y)*0.17-0.8,y*0.55]};
    var S=G3(cc,function(S,c){
      var a=rgbOf(c.c2),b=rgbOf(c.c1),i,j;
      for(i=0;i<N;i++)for(j=0;j<N;j++){var x0=-2.5+5*i/N,x1=-2.5+5*(i+1)/N,y0=-2.5+5*j/N,y1=-2.5+5*(j+1)/N,m=fx((x0+x1)/2,(y0+y1)/2),t=clamp(m/FM*1.4,0,1);
        var col="rgb("+Math.round(18+a[0]*t*.55+b[0]*t*t*.35)+","+Math.round(8+a[1]*t*.4+b[1]*t*t*.5)+","+Math.round(14+a[2]*t*.4+b[2]*t*t*.15)+")";
        S.quad([W(x0,y0),W(x1,y0),W(x1,y1),W(x0,y1)],col,"rgba(255,255,255,.07)",.92)}
      var cols=[c.c1,c.c2,c.c3,c.c4];
      O.forEach(function(q,k){for(var n=1;n<q.trail.length;n++){var A=q.trail[n-1],B=q.trail[n],pa=W(A[0],A[1]),pb=W(B[0],B[1]);pa[1]+=.04;pb[1]+=.04;S.line(pa,pb,cols[k],2.2,.95)}
        var p=W(q.p[0],q.p[1]);p[1]+=.07;S.dot(p,4.2,cols[k],1)});
      S.text([-1.45,-1.0,1.45],"w₁",c.muted,12);S.text([1.45,-1.0,-1.45],"w₂",c.muted,12);S.text([0,0.95,0],"loss ↑",c.muted,12)},{yaw:0.75,pitch:0.62,dist:5.2,spin:0.00012,tick:function(dt){if(!run)return;acc+=dt;while(acc>=110){step();acc-=110}O.forEach(function(q,k){$("#opL"+k).textContent=fx(q.p[0],q.p[1]).toFixed(3)})}});
    $("#opRun").addEventListener("click",function(){run=!run;this.textContent=run?"Pause":"Resume"});
    $("#opRestart").addEventListener("click",function(){reset();S.draw()});
    $("#opRand").addEventListener("click",function(){reset([-2.4+4.8*Math.random(),-2.4+4.8*Math.random()]);S.draw()});
    $("#opLr").addEventListener("input",function(){$("#opLrV").textContent=(+this.value).toFixed(2)});
    reset()})();

  /* initialisation */
  (function(){
    var cc=$("#inCv");if(!cc)return;var seed=3,data=null;
    function compute(){var act=$("#inAct").value,fn=act==="relu"?function(x){return Math.max(0,x)}:Math.tanh,n=128,Ls=30,res=[],stds=[0.02,Math.sqrt(1/n),Math.sqrt(2/n),0.2];
      stds.forEach(function(sd,k){var r=rng(seed*13+k),a=[],i,j,l,out=[];for(i=0;i<n;i++)a.push(gauss(r));
        for(l=0;l<=Ls;l++){var s=0;a.forEach(function(v){s+=v*v});out.push(Math.sqrt(s/n));var nx=[];for(j=0;j<n;j++){var z=0;for(i=0;i<n;i++)z+=gauss(r)*sd*a[i];nx.push(fn(z))}a=nx}res.push(out)});data=res}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,pad=36;g.clearRect(0,0,w,h);if(!data)return;var lo=-8,hi=3,X=function(l){return pad+l/30*(w-pad-8)},Y=function(v){var e=clamp(Math.log10(Math.max(v,1e-12)),lo,hi);return h-20-(e-lo)/(hi-lo)*(h-30)};
      g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";for(var e=lo;e<=hi;e+=3){var y=Y(Math.pow(10,e));g.beginPath();g.moveTo(pad,y);g.lineTo(w-6,y);g.stroke();g.fillText("1e"+e,2,y+3)}
      g.fillText("layer →",w-48,h-4);[c.c1,c.c3,c.c2,c.c4].forEach(function(col,k){g.strokeStyle=col;g.lineWidth=2.4;g.beginPath();data[k].forEach(function(v,l){l?g.lineTo(X(l),Y(v)):g.moveTo(X(l),Y(v))});g.stroke()})});
    function run(){compute();V.draw()}$("#inAct").addEventListener("change",run);$("#inNew").addEventListener("click",function(){seed=Math.floor(Math.random()*999);run()});run()})();

  /* dropout */
  (function(){
    var cc=$("#drCv");if(!cc)return;var L=[5,8,8,3],mask=[];
    function draw(){V.draw()}
    function resample(){var p=+$("#drP").value;mask=L.map(function(n,l){var a=[];for(var i=0;i<n;i++)a.push(l===0||l===L.length-1?1:(Math.random()>=p?1:0));return a});
      var act=0,tot=0;mask.forEach(function(m,l){if(l>0&&l<L.length-1)m.forEach(function(v){tot++;act+=v})});$("#drA").textContent=act+" of "+tot;$("#drS").textContent="× "+(1/(1-p)).toFixed(2);$("#drPV").textContent=p.toFixed(2)}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!mask.length)return;var pos=L.map(function(n,l){var a=[];for(var i=0;i<n;i++)a.push([w*(.1+.8*l/(L.length-1)),h*(.5+(i-(n-1)/2)*.105)]);return a});
      for(var l=0;l<L.length-1;l++)pos[l].forEach(function(a,i){pos[l+1].forEach(function(b,j){if(mask[l][i]&&mask[l+1][j]){g.strokeStyle=rgba(c.accent,.28);g.lineWidth=1;g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke()}})});
      pos.forEach(function(col,l){col.forEach(function(p,i){if(mask[l][i])dot(g,p[0],p[1],8,c.surface,l===L.length-1?c.c2:c.c1);else{g.globalAlpha=.35;dot(g,p[0],p[1],8,null,c.muted);g.strokeStyle=c.muted;g.beginPath();g.moveTo(p[0]-5,p[1]-5);g.lineTo(p[0]+5,p[1]+5);g.moveTo(p[0]+5,p[1]-5);g.lineTo(p[0]-5,p[1]+5);g.stroke();g.globalAlpha=1}})})});
    var tm=0;var Lp=loopOn(cc,function(dt){tm+=dt;if(tm>700){tm=0;resample();draw()}});if(!rm)Lp.set(true);
    $("#drP").addEventListener("input",function(){resample();draw()});resample();draw()})();

  /* lr schedules */
  (function(){
    var cc=$("#lrCv");if(!cc)return;var T=1000;
    function lr(k,t){var wu=100;
      if(k==="const")return 1;if(k==="step")return Math.pow(0.3,Math.floor(t/300));if(k==="cos")return 0.5*(1+Math.cos(Math.PI*t/T));
      if(k==="warm")return t<wu?t/wu:0.5*(1+Math.cos(Math.PI*(t-wu)/(T-wu)));return t<wu?t/wu:1-(t-wu)/(T-wu)}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,k=$("#lrSel").value,t=+$("#lrT").value,pad=30,X=function(s){return pad+s/T*(w-pad-8)},Y=function(v){return h-22-v*(h-34)};g.clearRect(0,0,w,h);
      g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";[0,.5,1].forEach(function(v){g.beginPath();g.moveTo(pad,Y(v));g.lineTo(w-6,Y(v));g.stroke();g.fillText(v.toFixed(1),2,Y(v)+3)});g.fillText("step →",w-44,h-6);
      g.strokeStyle=c.c1;g.lineWidth=2.6;g.beginPath();for(var s=0;s<=T;s+=5){s?g.lineTo(X(s),Y(lr(k,s))):g.moveTo(X(s),Y(lr(k,s)))}g.stroke();
      g.strokeStyle=c.ink;g.setLineDash([3,3]);g.beginPath();g.moveTo(X(t),8);g.lineTo(X(t),h-22);g.stroke();g.setLineDash([]);dot(g,X(t),Y(lr(k,t)),6,c.c2,c.bg)});
    function upd(){var t=+$("#lrT").value;$("#lrTV").textContent=t;$("#lrV").textContent="× "+lr($("#lrSel").value,t).toFixed(3)+" of peak";V.draw()}
    $("#lrSel").addEventListener("change",upd);$("#lrT").addEventListener("input",upd);upd()})();
})();
