/* Transformer and LLM demos */
(function(){
  function fmtN(x){return x>=1e12?(x/1e12).toFixed(2)+" T":x>=1e9?(x/1e9).toFixed(2)+" B":x>=1e6?(x/1e6).toFixed(2)+" M":x>=1e3?(x/1e3).toFixed(1)+" k":String(Math.round(x))}
  function chips(el,arr,cls){el.innerHTML=arr.map(function(t){return '<span class="tok"'+(cls?' style="'+cls+'"':"")+'>'+esc(t)+'</span>'}).join("")}

  /* BPE */
  (function(){
    if(!$("#bpeTxt"))return;
    function train(text,n){var words=text.toLowerCase().split(/\s+/).filter(Boolean),freq={},seqs={},merges=[];words.forEach(function(w){freq[w]=(freq[w]||0)+1});Object.keys(freq).forEach(function(w){seqs[w]=("▁"+w).split("")});
      for(var m=0;m<n;m++){var pc={};Object.keys(seqs).forEach(function(w){var s=seqs[w];for(var i=0;i<s.length-1;i++){var k=s[i]+"\u0001"+s[i+1];pc[k]=(pc[k]||0)+freq[w]}});
        var best=null;Object.keys(pc).forEach(function(k){if(!best||pc[k]>pc[best]||(pc[k]===pc[best]&&k<best))best=k});if(!best||pc[best]<2)break;
        var ab=best.split("\u0001");merges.push([ab[0],ab[1],pc[best]]);Object.keys(seqs).forEach(function(w){var s=seqs[w],o=[],i=0;while(i<s.length){if(i<s.length-1&&s[i]===ab[0]&&s[i+1]===ab[1]){o.push(ab[0]+ab[1]);i+=2}else{o.push(s[i]);i++}}seqs[w]=o})}
      return {seqs:seqs,merges:merges,freq:freq}}
    function apply(word,merges){var s=("▁"+word.toLowerCase()).split("");merges.forEach(function(m){var o=[],i=0;while(i<s.length){if(i<s.length-1&&s[i]===m[0]&&s[i+1]===m[1]){o.push(m[0]+m[1]);i+=2}else{o.push(s[i]);i++}}s=o});return s}
    function upd(){var n=+$("#bpeN").value;$("#bpeNV").textContent=n;var r=train($("#bpeTxt").value,n),words=Object.keys(r.seqs);
      chips($("#bpeMerges"),r.merges.map(function(m,i){return (i+1)+". "+m[0]+" + "+m[1]+" → "+m[0]+m[1]+" ("+m[2]+")"}));if(!r.merges.length)$("#bpeMerges").innerHTML='<span class="note">No merges yet. Slide right.</span>';
      var seg=$("#bpeSeg");seg.innerHTML=words.map(function(w){return '<span style="display:inline-flex;gap:3px;margin-right:10px;flex-wrap:wrap">'+r.seqs[w].map(function(t){return '<span class="tok">'+esc(t)+'</span>'}).join("")+'</span>'}).join("");
      var chars={};words.forEach(function(w){("▁"+w).split("").forEach(function(c){chars[c]=1})});$("#bpeV").textContent=Object.keys(chars).length+r.merges.length;
      var tot=0,cnt=0;words.forEach(function(w){tot+=r.seqs[w].length*r.freq[w];cnt+=r.freq[w]});$("#bpeR").textContent=cnt?(tot/cnt).toFixed(2):"-";
      var ws=$("#bpeIn").value.split(/\s+/).filter(Boolean),o=$("#bpeOut");o.innerHTML=ws.map(function(w){return '<span style="display:inline-flex;gap:3px;margin-right:10px;flex-wrap:wrap">'+apply(w,r.merges).map(function(t){return '<span class="tok" style="border-color:var(--accent)">'+esc(t)+'</span>'}).join("")+'</span>'}).join("")}
    ["bpeTxt","bpeN","bpeIn"].forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* embeddings 3D */
  (function(){
    var cc=$("#embCv");if(!cc)return;
    var W={man:[-0.2,-0.7,0],woman:[-0.2,0.7,0],king:[1,-0.7,0],queen:[1,0.7,0],prince:[0.55,-0.7,0.5],princess:[0.55,0.7,0.5],France:[-1.4,-0.9,-1.3],Paris:[-0.6,-0.5,-1.3],Japan:[-1.4,0.1,-1.3],Tokyo:[-0.6,0.5,-1.3],Italy:[-1.4,1.0,-1.3],Rome:[-0.6,1.4,-1.3],cat:[0.5,-0.6,1.6],dog:[0.7,-0.2,1.5],lion:[1.1,-0.6,1.9],tiger:[1.2,-0.3,2.0]};
    var A=[["man","king","woman"],["France","Paris","Japan"],["Italy","Rome","France"],["king","queen","prince"]];
    function solve(i){var a=W[A[i][0]],b=W[A[i][1]],c=W[A[i][2]],q=[c[0]+b[0]-a[0],c[1]+b[1]-a[1],c[2]+b[2]-a[2]],best=null,bd=1e9;
      Object.keys(W).forEach(function(w){if(A[i].indexOf(w)>=0)return;var v=W[w],d=Math.pow(v[0]-q[0],2)+Math.pow(v[1]-q[1],2)+Math.pow(v[2]-q[2],2);if(d<bd){bd=d;best=w}});return {q:q,w:best}}
    function arrow(S,p,q,col,wd){S.line(p,q,col,wd,1);var d=[q[0]-p[0],q[1]-p[1],q[2]-p[2]],L=Math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])||1;S.dot(q,3,col,1)}
    var S=G3(cc,function(S,c){var i=+$("#embSel").value,r=solve(i),yv=function(w){return W[w]};
      Object.keys(W).forEach(function(w){var p=W[w],col=w==="man"||w==="woman"||w==="king"||w==="queen"||w==="prince"||w==="princess"?c.c1:(["cat","dog","lion","tiger"].indexOf(w)>=0?c.c3:c.c4);S.dot([p[0],p[1],p[2]],4,col,1);S.text([p[0],p[1]+0.14,p[2]],w,w===r.w?c.ink:c.muted,w===r.w?13:11)});
      arrow(S,yv(A[i][0]),yv(A[i][1]),c.c2,2.2);arrow(S,yv(A[i][2]),W[r.w],c.c2,2.2);S.dot(r.q,6,c.ink,.9);
      S.line([-1.6,-1.2,-1.2],[1.6,-1.2,-1.2],c.line,1,.6);
      var e=$("#embEq"),a=A[i];e.textContent=a[1]+" − "+a[0]+" + "+a[2];$("#embAns").textContent=r.w},{yaw:0.55,pitch:0.28,dist:5.2,spin:0.00012});
    $("#embSel").addEventListener("change",function(){S.draw()})})();

  /* attention heatmap */
  (function(){
    var cc=$("#atCv");if(!cc)return;var sel=-1,scale=true,mask=false,D=8;
    function hash(s){var h=2166136261;for(var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
    var rq=rng(11),Wq=[],Wk=[];for(var i=0;i<D;i++){Wq.push([]);Wk.push([]);for(var j=0;j<D;j++){Wq[i].push(gauss(rq)/Math.sqrt(D)*1.6);Wk[i].push(gauss(rq)/Math.sqrt(D)*1.6)}}
    function proj(x,W){var o=[];for(var j=0;j<D;j++){var s=0;for(var i=0;i<D;i++)s+=x[i]*W[i][j];o.push(s)}return o}
    function compute(){var toks=$("#atS").value.split(" "),T=+$("#atT").value,E=toks.map(function(t,p){var r=rng(hash(t)),v=[];for(var d=0;d<D;d++)v.push(gauss(r)+0.9*(d%2?Math.cos(p/Math.pow(10,d/D*2)):Math.sin(p/Math.pow(10,d/D*2))));return v}),
      Q=E.map(function(e){return proj(e,Wq)}),K=E.map(function(e){return proj(e,Wk)}),M=[];
      Q.forEach(function(q,i){var row=toks.map(function(_,j){if(mask&&j>i)return -1e9;var s=0;for(var d=0;d<D;d++)s+=q[d]*K[j][d];return s/(scale?Math.sqrt(D):1)/T}),mx=Math.max.apply(null,row),ex=row.map(function(v){return Math.exp(v-mx)}),sm=ex.reduce(function(a,b){return a+b},0);M.push(ex.map(function(v){return v/sm}))});
      return {toks:toks,M:M}}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,R=compute(),n=R.toks.length,ml=84,mt=78,cs=Math.min((w-ml-6)/n,(h-mt-6)/n);g.clearRect(0,0,w,h);
      g.font="12px "+getComputedStyle(document.body).fontFamily;
      R.toks.forEach(function(t,i){g.fillStyle=i===sel?c.c1:c.ink;g.textAlign="right";g.fillText(t,ml-8,mt+i*cs+cs/2+4);g.save();g.translate(ml+i*cs+cs/2+4,mt-8);g.rotate(-Math.PI/3);g.textAlign="left";g.fillStyle=c.ink;g.fillText(t,0,0);g.restore()});g.textAlign="left";
      var rgb=rgbOf(c.c1);R.M.forEach(function(row,i){row.forEach(function(p,j){g.fillStyle="rgba("+rgb[0]+","+rgb[1]+","+rgb[2]+","+(0.04+0.96*Math.min(1,p*1.8))+")";g.fillRect(ml+j*cs+1,mt+i*cs+1,cs-2,cs-2);if(cs>34&&p>0.01){g.fillStyle=p>0.35?"#000":c.muted;g.font="10px sans-serif";g.textAlign="center";g.fillText(Math.round(p*100),ml+j*cs+cs/2,mt+i*cs+cs/2+3);g.textAlign="left";g.font="12px "+getComputedStyle(document.body).fontFamily}})});
      if(sel>=0){g.strokeStyle=c.c2;g.lineWidth=2;g.strokeRect(ml,mt+sel*cs,cs*n,cs)}
      o.cs=cs;o.ml=ml;o.mt=mt;o.n=n;o.R=R});
    function note(){var R=compute();if(sel<0||sel>=R.toks.length){$("#atNote").textContent="Click a row to see where that token attends.";return}var row=R.M[sel].map(function(p,j){return [p,R.toks[j]]}).sort(function(a,b){return b[0]-a[0]}).slice(0,3);$("#atNote").textContent="“"+R.toks[sel]+"” attends most to: "+row.map(function(r){return "“"+r[1]+"” "+Math.round(r[0]*100)+"%"}).join(", ")+"."}
    cc.addEventListener("pointerdown",function(e){var m=mouse(cc,e);if(!V.n&&!cc.__o){}var r=cc.getBoundingClientRect();var n=compute().toks.length,cs=Math.min((r.width-84-6)/n,(r.height-78-6)/n),i=Math.floor((m.y-78)/cs);sel=i>=0&&i<n?i:-1;V.draw();note()});
    function re(){sel=-1;V.draw();note()}
    $("#atS").addEventListener("change",re);$("#atT").addEventListener("input",function(){$("#atTV").textContent=(+this.value).toFixed(1);V.draw();note()});
    $("#atScale").addEventListener("click",function(){scale=!scale;this.setAttribute("aria-pressed",scale);this.textContent="Divide by √d "+(scale?"on":"off");V.draw();note()});
    $("#atMask").addEventListener("click",function(){mask=!mask;this.setAttribute("aria-pressed",mask);this.textContent="Causal mask "+(mask?"on":"off");V.draw();note()});
    V.draw()})();

  /* transformer 3D */
  (function(){
    var cc=$("#trCv3");if(!cc)return;var sel="all";
    var INFO={all:["Overview","Tokens enter at the bottom, flow up the residual stream through repeated attention and MLP blocks, and leave as next-token probabilities.","P(next token) = softmax( Unembed( x_L ) )"],
      emb:["Embedding + position","Each token ID looks up a vector, and position information is added so the model knows order.","x₀ = E[token] + position"],
      attn:["Multi-head attention","Every token mixes in information from earlier tokens. Each head learns its own pattern, and the heads' outputs are concatenated and projected.","softmax(Q·Kᵀ / √d) · V, per head"],
      mlp:["Feed-forward (MLP)","Applied to each token independently. It expands to about four times the width, applies a nonlinearity and projects back. Much of the stored knowledge lives here.","W₂ · GELU(W₁ · x)"],
      res:["Residual stream and normalisation","The stream carries each token's running vector. Layers add their output to it rather than replace it, and LayerNorm rescales the input to each sublayer.","x ← x + F( LayerNorm(x) )"],
      out:["Output (unembedding)","A final norm and a linear map turn the last vector into one score per vocabulary token. Softmax gives the next-token distribution.","logits = LN(x_L) · Eᵀ"]};
    var words=["The","cat","sat","on","the"];
    function info(){var i=INFO[sel];$("#trInfo").innerHTML="<h4>"+esc(i[0])+"</h4><p>"+esc(i[1])+'</p><div class="math">'+esc(i[2])+"</div>"}
    var S=G3(cc,function(S,c){var L=+$("#trL").value,H=+$("#trH").value,T=5,U=2*L+2,Hg=3.2,dy=Hg/U,y0=-Hg/2,zs=function(i){return (i-2)*0.62},t=S.t*0.00035;
      function lv(k){return y0+(k+0.5)*dy}
      function slab(k,col,key,xw,label){var on=sel==="all"||sel===key,y=lv(k);S.box(0,y,0,xw,dy*0.5,3.3,rgba(col,on?(sel===key?.42:.2):.05),on?col:rgba(col,.25),on?1:.4);if(label)S.text([xw/2+0.15,y,1.9],label,on?c.ink:c.muted,11,"left")}
      slab(0,c.ink,"emb",1.3,"Embedding + position");
      for(var l=0;l<L;l++){slab(1+2*l,c.c2,"attn",1.3,l===0||L<=4?"Attention · "+H+" head"+(H>1?"s":""):(l===L-1?"…":""));slab(2+2*l,c.c1,"mlp",1.8,l===0||L<=4?"MLP":"")}
      slab(U-1,c.ink,"out",1.3,"Unembedding → logits");
      /* residual lines and norms */
      var rc=sel==="res"?c.accent:c.muted;for(var i=0;i<T;i++){S.line([-0.95,y0,zs(i)],[-0.95,y0+Hg,zs(i)],rc,sel==="res"?2.4:1.2,sel==="res"||sel==="all"?.9:.35)}
      for(l=0;l<L;l++)for(var j=0;j<2;j++){var yy=lv(1+2*l+j)-dy*0.42;if(sel==="res"||sel==="all")S.ring([-0.95,yy,0],0.22,14,sel==="res"?c.accent:c.muted,1.2,.9,"x")}
      /* attention edges among tokens */
      for(l=0;l<L;l++){var yy2=lv(1+2*l),on=sel==="all"||sel==="attn";for(var h=0;h<H;h++){var x=-0.5+(H===1?0.5:h/(H-1))*1.0;for(var a=0;a<T;a++)for(var b=0;b<a;b++){var s=Math.abs(Math.sin(a*3.1+b*1.7+h*2.3+l));S.line([x,yy2,zs(a)],[x,yy2,zs(b)],c.c2,1,on?0.08+0.38*s*(sel==="attn"?1.6:1):.04)}}}
      /* mlp inner dots */
      for(l=0;l<L;l++){var ym=lv(2+2*l),on2=sel==="all"||sel==="mlp";for(i=0;i<T;i++)for(var k=-1;k<=1;k++)S.dot([k*0.5,ym,zs(i)],1.4,c.c1,on2?.9:.25)}
      /* pulses and tokens */
      for(i=0;i<T;i++){var p=((t+i*0.19)%1);S.dot([-0.95,y0+p*Hg,zs(i)],3.6,c.c1,1);S.text([0,y0-0.2,zs(i)],words[i],c.ink,11)}
      S.text([0,y0+Hg+0.2,zs(4)],"mat ?",c.c2,12)},{yaw:0.85,pitch:0.28,dist:6,spin:0.00014,zoom:0.74});
    var seg=$("#trSeg");seg.addEventListener("click",function(e){var b=e.target.closest("button");if(!b)return;sel=b.dataset.k;$$("button",seg).forEach(function(x){x.setAttribute("aria-pressed",x===b?"true":"false")});info();S.draw()});
    ["trL","trH"].forEach(function(id){$("#"+id).addEventListener("input",function(){$("#trLV").textContent=$("#trL").value;$("#trHV").textContent=$("#trH").value;S.draw()})});
    info();
    /* parameter counter */
    function pm(){var L=+$("#pmL").value,d=+$("#pmD").value,V=+$("#pmV").value,a=4*L*d*d,m=8*L*d*d,e=V*d;$("#pmA").textContent=fmtN(a);$("#pmM").textContent=fmtN(m);$("#pmE").textContent=fmtN(e);$("#pmT").textContent=fmtN(a+m+e);$("#pmG").textContent=((a+m+e)*2/1e9).toFixed(1)+" GB"}
    ["pmL","pmD","pmV"].forEach(function(id){$("#"+id).addEventListener("input",pm)});pm()})();

  /* sampling */
  (function(){
    var box=$("#spBars");if(!box)return;
    var P={"The capital of France is":[[" Paris",8],[" the",4.9],[" a",4.3],[" located",3.9],[" known",3.6],[" Lyon",3.4],[" France",3.1],[",",2.9],[" now",2.4],[" one",2.2],[" called",2],["\n",1.8]],
      "Once upon a":[[" time",9],[" midnight",5.2],[" hill",4.4],[" day",4.2],[" dream",3.9],[" star",3.5],[" long",3.2],[" winter",3],[" river",2.7],[" journey",2.5],[" shore",2.3],[" mountain",2]],
      "The best way to learn is":[[" by",6.5],[" to",5.9],[" through",5.5],[" practice",5.1],[" doing",4.8],[" with",4.5],[" teaching",4.2],[" experimenting",3.9],[" reading",3.7],[" asking",3.4],[" building",3.2],[" repeating",2.9]]};
    var sp=$("#spP");sp.innerHTML=Object.keys(P).map(function(k){return "<option>"+esc(k)+"</option>"}).join("");
    var out="",cur=null;
    box.innerHTML=Array(12).fill(0).map(function(_,i){return '<div class="bar-r" id="spr'+i+'"><span class="nm"></span><div class="t"><i></i></div><span class="pc"></span></div>'}).join("");
    function calc(){var d=P[sp.value],T=+$("#spT").value,K=+$("#spK").value,Pp=+$("#spPp").value;$("#spTV").textContent=T.toFixed(1);$("#spKV").textContent=K;$("#spPV").textContent=Pp.toFixed(2);
      var mx=d[0][1],e=d.map(function(x){return Math.exp((x[1]-mx)/T)}),s=e.reduce(function(a,b){return a+b},0),p=e.map(function(v){return v/s}),orig=p.slice();
      var keep=p.map(function(_,i){return i<K});var s2=0;p.forEach(function(v,i){if(keep[i])s2+=v});p=p.map(function(v,i){return keep[i]?v/s2:0});
      var cum=0;p.forEach(function(v,i){if(!keep[i])return;if(cum>=Pp)keep[i]=false;else cum+=v});var s3=0;p.forEach(function(v,i){if(keep[i])s3+=v});p=p.map(function(v,i){return keep[i]?v/s3:0});
      cur={p:p,d:d};p.forEach(function(v,i){var r=$("#spr"+i),on=keep[i];$(".nm",r).textContent=JSON.stringify(d[i][0]).slice(1,-1);var b=$("i",r);b.style.width=(on?v:orig[i])*100+"%";b.className=on?"":"off";b.style.background=on?(i===0?"var(--c2)":"var(--c1)"):"";$(".pc",r).textContent=on?(v*100).toFixed(1)+"%":"cut"})}
    ["spT","spK","spPp"].forEach(function(id){$("#"+id).addEventListener("input",calc)});sp.addEventListener("change",function(){out="";calc();$("#spOut").textContent="Prompt + sampled tokens appear here."});
    $("#spGo").addEventListener("click",function(){var r=Math.random(),c=0,i=0;for(;i<cur.p.length;i++){c+=cur.p[i];if(r<=c)break}i=Math.min(i,cur.p.length-1);out+=cur.d[i][0];$("#spOut").textContent=sp.value+out});
    $("#spClr").addEventListener("click",function(){out="";$("#spOut").textContent="Prompt + sampled tokens appear here."});calc()})();

  /* scale */
  (function(){
    if(!$("#scN"))return;
    function upd(){var N=+$("#scN").value*1e9,D=+$("#scD").value*1e9,F=+$("#scF").value*1e12,u=+$("#scU").value/100;$("#scUV").textContent=Math.round(u*100)+"%";var C=6*N*D,sec=C/(F*u),hrs=sec/3600;
      $("#scR").textContent=(D/N).toFixed(0)+(D/N>=20?" (above 20)":" (below 20)");$("#scC").textContent=C.toExponential(2)+" FLOPs";$("#scH").textContent=hrs>=1e6?(hrs/1e6).toFixed(2)+" M":Math.round(hrs).toLocaleString();var days=hrs/1000/24;$("#scDay").textContent=days>=1?days.toFixed(1)+" days":(days*24).toFixed(1)+" hours"}
    ["scN","scD","scF","scU"].forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* LoRA */
  (function(){
    var cc=$("#lrFig");if(!cc)return;
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,d=+$("#lrD").value,r=+$("#lrR").value,S=Math.min(h-30,w/4),rw=Math.max(4,S*r/d);g.clearRect(0,0,w,h);var cx=w/2-S*1.4,y=15;
      g.fillStyle=rgba(c.c1,.22);g.strokeStyle=c.c1;g.lineWidth=2;g.fillRect(cx,y,S,S);g.strokeRect(cx,y,S,S);g.fillStyle=c.ink;g.font="12px sans-serif";g.textAlign="center";g.fillText("W  d×d (frozen)",cx+S/2,y+S/2);
      g.fillText("+",cx+S+S*0.2,y+S/2);var bx=cx+S+S*0.4;g.fillStyle=rgba(c.c2,.4);g.strokeStyle=c.c2;g.fillRect(bx,y,rw,S);g.strokeRect(bx,y,rw,S);g.fillText("B",bx+rw/2,y+S+14);
      var ax=bx+rw+S*0.15;g.fillRect(ax,y+S/2-rw/2,S,rw);g.strokeRect(ax,y+S/2-rw/2,S,rw);g.fillStyle=c.ink;g.fillText("A",ax+S/2,y+S/2+rw/2+14);g.textAlign="left"});
    function upd(){var d=+$("#lrD").value,r=+$("#lrR").value;$("#lrRV").textContent=r;var f=d*d,l=2*d*r;$("#lrF").textContent=fmtN(f);$("#lrL").textContent=fmtN(l);$("#lrP").textContent=(100*l/f).toFixed(2)+"%";V.draw()}
    ["lrD","lrR"].forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* retrieval */
  (function(){
    var box=$("#rgBars");if(!box)return;
    var DOCS=["Gradient descent updates weights by following the negative gradient of the loss.","Transformers use self-attention to relate every token to every other token.","Convolutional networks slide filters over images to detect edges and textures.","Random forests average many decision trees trained on bootstrap samples.","Reinforcement learning agents learn a policy by maximising cumulative reward.","Diffusion models generate images by gradually removing noise from random samples.","LoRA fine-tunes large models by training small low-rank adapter matrices.","Dropout randomly disables neurons during training to reduce overfitting."];
    var STOP="the a an of to by is are and in over do how from can with use uses on for it that this".split(" ");
    function tok(s){return s.toLowerCase().split(/[^a-z]+/).filter(function(w){return w&&STOP.indexOf(w)<0}).map(function(w){return w.replace(/(ing|ed|es|s)$/,"")})}
    var T=DOCS.map(tok),df={};T.forEach(function(t){var u={};t.forEach(function(w){u[w]=1});Object.keys(u).forEach(function(w){df[w]=(df[w]||0)+1})});
    function vec(t){var v={};t.forEach(function(w){v[w]=(v[w]||0)+1});Object.keys(v).forEach(function(w){v[w]*=Math.log((DOCS.length+1)/((df[w]||0)+1))+1});return v}
    function cos(a,b){var s=0,na=0,nb=0;Object.keys(a).forEach(function(k){na+=a[k]*a[k];if(b[k])s+=a[k]*b[k]});Object.keys(b).forEach(function(k){nb+=b[k]*b[k]});return na&&nb?s/Math.sqrt(na*nb):0}
    var DV=T.map(vec);box.innerHTML=DOCS.map(function(_,i){return '<div class="bar-r" style="grid-template-columns:44px minmax(0,1fr) 54px" id="rgr'+i+'"><span>#'+(i+1)+'</span><div class="t" style="height:auto;padding:3px 8px;min-height:22px;position:relative"><i style="position:absolute;left:0;top:0;bottom:0;opacity:.35"></i><span class="tx" style="position:relative;font:400 .82rem var(--font-body)"></span></div><span class="pc"></span></div>'}).join("");
    function run(){var q=vec(tok($("#rgQ").value)),sc=DV.map(function(v,i){return [cos(q,v),i]}).sort(function(a,b){return b[0]-a[0]}),top=sc.slice(0,2).filter(function(s){return s[0]>0});
      DOCS.forEach(function(d,i){var r=$("#rgr"+i),s=sc.filter(function(x){return x[1]===i})[0][0];$(".tx",r).textContent=d;$("i",r).style.width=(s*100)+"%";$("i",r).style.background=top.some(function(t){return t[1]===i})?"var(--c2)":"var(--c1)";$(".pc",r).textContent=s.toFixed(2)});
      $("#rgPrompt").textContent=top.length?"Answer using only this context:\n"+top.map(function(t,k){return "["+(k+1)+"] "+DOCS[t[1]]}).join("\n")+"\n\nQuestion: "+$("#rgQ").value:"No chunk matched the question. A real system would say it cannot answer, instead of letting the model guess."}
    $("#rgQ").addEventListener("input",run);run()})();
})();
autoRun('spGo');
