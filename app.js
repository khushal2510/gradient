/* Applied LLMs page */
(function(){
  function sig(x){return 1/(1+Math.exp(-x))}
  function lab(g,t,x,y,col,al){g.fillStyle=col;g.textAlign=al||"left";g.fillText(t,x,y)}

  /* chunking */
  (function(){
    var box=$("#chBox");if(!box)return;
    var TXT="Fine-tuning adapts a pretrained model to a task. Full fine-tuning updates every weight, which needs a lot of memory for gradients and optimiser states. LoRA freezes the base weights and trains small low-rank matrices instead, so the adapter file is tiny and many adapters can share one base model. QLoRA goes further by storing the frozen base in four bits. After training you can keep the adapter separate or merge it into the base weights for deployment. Evaluate on held-out data and on general benchmarks to check that new skills did not erase old ones.".split(/\s+/);
    function draw(){var s=+$("#chS").value,o=Math.min(+$("#chO").value,s-1),i=0,out=[],step=s-o,total=0;
      for(var st=0;st<TXT.length;st+=step){var ch=TXT.slice(st,st+s);if(!ch.length)break;total+=ch.length;
        var ov=st>0?o:0,h=(ov?"<mark>"+esc(ch.slice(0,ov).join(" "))+"</mark> ":"")+esc(ch.slice(ov).join(" "));out.push('<div class="chunk"><b>'+(++i)+"</b>"+h+"</div>");if(st+s>=TXT.length)break}
      box.innerHTML=out.join("");$("#chSV").textContent=s;$("#chOV").textContent=o;$("#chN").textContent=i;$("#chW").textContent=total;$("#chX").textContent="+"+Math.round((total/TXT.length-1)*100)+"%"}
    $("#chS").addEventListener("input",draw);$("#chO").addEventListener("input",draw);draw()})();

  /* retrieval comparison */
  (function(){
    var box=$("#rgBox");if(!box)return;
    var D=["LoRA fine-tunes a model by training small low-rank matrices while the base weights stay frozen, which cuts memory a lot.","The KV cache stores keys and values of earlier tokens so each new token does not recompute attention over the whole prompt.","BM25 is a keyword ranking function that scores documents by term frequency and rarity, and it still powers most search engines.","Dense embeddings map text to vectors so that similar meanings are close, even when the exact words differ.","A reranker is a cross-encoder that reads the query and a candidate passage together to give a more accurate relevance score.","Reciprocal rank fusion merges ranked lists from several retrievers by summing one over a constant plus the rank.","Gradient clipping rescales gradients whose norm is too large, which prevents exploding updates in transformers and RNNs.","Quantisation stores weights in fewer bits such as int8 or int4, which shrinks memory and speeds up inference.","Chunking splits long documents into passages. Overlap keeps sentences that straddle a boundary retrievable.","Prompt injection hides instructions inside retrieved text or user input so the model ignores its real task.","Continuous batching lets a server add new requests to a running batch as soon as another request finishes.","Vector databases use approximate nearest neighbour indexes such as HNSW to search millions of embeddings quickly."];
    var STOP={a:1,an:1,the:1,of:1,to:1,in:1,is:1,it:1,and:1,or:1,for:1,by:1,so:1,as:1,that:1,this:1,with:1,from:1,on:1,at:1,be:1,can:1,way:1,two:1,do:1,not:1,each:1,all:1,while:1,which:1,when:1,even:1,more:1,than:1,such:1,are:1,its:1};
    function stem(w){return w.length>4?w.replace(/(ing|es|ed|s)$/,""):w}
    function toks(s){return s.toLowerCase().replace(/[^a-z0-9]/g," ").split(/\s+/).filter(function(w){return w&&!STOP[w]}).map(stem)}
    var T=D.map(toks),avg=T.reduce(function(a,t){return a+t.length},0)/T.length,df={};T.forEach(function(t){var u={};t.forEach(function(w){u[w]=1});for(var w in u)df[w]=(df[w]||0)+1});
    function bm25(q){var qt=toks(q);return T.map(function(t){var s=0;qt.forEach(function(w){var f=t.filter(function(x){return x===w}).length;if(!f)return;var idf=Math.log(1+(D.length-(df[w]||0)+.5)/((df[w]||0)+.5));s+=idf*f*2.5/(f+1.5*(.25+.75*t.length/avg))});return s})}
    function tri(s){var v={},t="  "+s.toLowerCase().replace(/[^a-z0-9 ]/g," ")+" ";for(var i=0;i<t.length-2;i++){var g=t.substr(i,3);if(g.trim().length<3&&g.indexOf(" ")>=0&&g.trim().length<2)continue;v[g]=(v[g]||0)+1}return v}
    function cos(a,b){var d=0,na=0,nb=0,k;for(k in a){na+=a[k]*a[k];if(b[k])d+=a[k]*b[k]}for(k in b)nb+=b[k]*b[k];return na&&nb?d/Math.sqrt(na*nb):0}
    var V=D.map(tri);
    function rank(sc){return sc.map(function(s,i){return [i,s]}).sort(function(a,b){return b[1]-a[1]}).map(function(p){return p[0]})}
    function draw(){var q=$("#rgQ").value,s1=bm25(q),qv=tri(q),s2=V.map(function(v){return cos(qv,v)}),r1=rank(s1),r2=rank(s2),f={};
      [r1,r2].forEach(function(r,k){r.forEach(function(d,i){if((k===0?s1:s2)[d]>0)f[d]=(f[d]||0)+1/(60+i+1)})});
      var hy=Object.keys(f).map(Number).sort(function(a,b){return f[b]-f[a]}),qs=toks(q).map(function(w){return w.slice(0,4)});
      var rr=hy.slice(0,6).map(function(d){var t=D[d].toLowerCase(),cov=qs.filter(function(w){return t.indexOf(w)>=0}).length/(qs.length||1);return [d,f[d]*60+cov]}).sort(function(a,b){return b[1]-a[1]}).map(function(p){return p[0]});
      var cols=[["BM25 keywords",r1,s1],["Vector (trigram stand-in)",r2,s2],["Hybrid (RRF)",hy,null],["Hybrid + toy rerank",rr,null]];
      box.innerHTML=cols.map(function(c){return '<div class="col"><h4>'+c[0]+"</h4>"+c[1].slice(0,3).filter(function(d,i){return !c[2]||c[2][d]>0}).map(function(d,i){return '<div class="hit"><b>'+(i+1)+"</b>"+esc(D[d].slice(0,86))+(D[d].length>86?"…":"")+"</div>"}).join("")+"</div>"}).join("")}
    $("#rgQ").addEventListener("input",draw);$("#rgSel").addEventListener("change",function(){$("#rgQ").value=this.value;draw()});draw()})();

  /* agent loop */
  (function(){
    var log=$("#agLog");if(!log)return;
    function calc(e){if(!/^[0-9+\-*\/(). ]+$/.test(e))return "error";try{return +Function('"use strict";return ('+e+")")().toFixed(4)/1}catch(x){return "error"}}
    var SC=[[["model","I will use the calculator."],["call",'{"tool":"calculator","args":{"expression":"0.17*2340"}}'],["tool",function(){return '{"result": '+calc("0.17*2340")+"}"}],["model","17% of 2,340 is 397.8."]],
      [["model","I need the current temperature. Calling the weather tool."],["call",'{"tool":"get_weather","args":{"city":"Pune"}}'],["tool",function(){return '{"error":"timeout"}'}],["model","The tool timed out. Retrying once."],["call",'{"tool":"get_weather","args":{"city":"Pune"}}'],["tool",function(){return '{"temp_c": 31}'}],["model","It is 31°C in Pune, so yes, it is above 30°C."]],
      [["model","First I need the order subtotal."],["call",'{"tool":"get_order","args":{"order_id":8841}}'],["tool",function(){return '{"subtotal": 1200}'}],["call",'{"tool":"calculator","args":{"expression":"1200*1.18"}}'],["tool",function(){return '{"result": '+calc("1200*1.18")+"}"}],["model","The total with 18% tax is 1,416."]]];
    var i=0,calls=0,stopped=false;
    function render(){var sc=SC[+$("#agQ").value],h="";calls=0;stopped=false;for(var k=0;k<i;k++){var s=sc[k];if(s[0]==="call"){calls++;if(calls>+$("#agL").value){stopped=true;h+='<div class="msg stop"><b>limit</b>Stopped: step limit reached before an answer.</div>';break}}
        h+='<div class="msg '+s[0]+'"><b>'+({model:"model",call:"tool call",tool:"result"})[s[0]]+"</b><code>"+esc(typeof s[1]==="function"?s[1]():s[1])+"</code></div>"}
      log.innerHTML=h||'<p class="note">Press Step to begin.</p>';$("#agLV").textContent=$("#agL").value;$("#agStep").disabled=stopped||i>=sc.length}
    $("#agStep").addEventListener("click",function(){var sc=SC[+$("#agQ").value];if(i<sc.length){i++;render()}});
    $("#agRe").addEventListener("click",function(){i=0;render()});$("#agQ").addEventListener("change",function(){i=0;render()});$("#agL").addEventListener("input",render);render()})();

  /* JSON validator */
  (function(){
    var T=$("#jsT");if(!T)return;var BAD='{"intent": "complaint", "priority": "high", "tags": ["billing", "late", "refund", "urgent"], "notes": "please help"}',GOOD='{"intent": "refund", "priority": 2, "tags": ["billing", "late"]}';
    function val(){var s=T.value,o,E=[];try{o=JSON.parse(s)}catch(e){$("#jsOut").innerHTML="<h4>Not valid JSON</h4><ul><li>"+esc(e.message)+"</li></ul><p class='note'>With constrained decoding this cannot happen. Without it, retry with the error message.</p>";return}
      if(o===null||typeof o!=="object"||Array.isArray(o)){E.push("Top level must be an object")}else{
        ["intent","priority","tags"].forEach(function(k){if(!(k in o))E.push("Missing required field: "+k)});
        if("intent" in o&&["refund","bug","question"].indexOf(o.intent)<0)E.push("intent must be one of refund, bug, question (got "+JSON.stringify(o.intent)+")");
        if("priority" in o&&!(Number.isInteger(o.priority)&&o.priority>=1&&o.priority<=5))E.push("priority must be an integer from 1 to 5 (got "+JSON.stringify(o.priority)+")");
        if("tags" in o){if(!Array.isArray(o.tags))E.push("tags must be an array");else{if(o.tags.length>3)E.push("tags has "+o.tags.length+" items, maximum is 3");if(o.tags.some(function(t){return typeof t!=="string"}))E.push("every tag must be a string")}}
        Object.keys(o).forEach(function(k){if(["intent","priority","tags"].indexOf(k)<0)E.push("Unexpected field: "+k)})}
      $("#jsOut").innerHTML=E.length?"<h4>"+E.length+" problem"+(E.length>1?"s":"")+"</h4><ul>"+E.map(function(x){return "<li>"+esc(x)+"</li>"}).join("")+"</ul>":"<h4>Valid</h4><p class='note'>Matches the schema. Business rules still need checking in code.</p>"}
    T.value=BAD;T.addEventListener("input",val);$("#jsBad").addEventListener("click",function(){T.value=BAD;val()});$("#jsGood").addEventListener("click",function(){T.value=GOOD;val()});val()})();

  /* self-consistency */
  (function(){
    var cc=$("#appRs");if(!cc)return;var cache={},NM=32,TR=1500;
    function curve(p,K){var key=p+"|"+K;if(cache[key])return cache[key];var r=rng(9),hit=new Float64Array(NM+1),t,n,k;
      for(t=0;t<TR;t++){var cnt=new Int16Array(K+1);for(n=1;n<=NM;n++){var a=r()<p?0:1+Math.floor(r()*K);cnt[a]++;var mx=0,ties=0;for(k=0;k<=K;k++){if(cnt[k]>mx){mx=cnt[k];ties=1}else if(cnt[k]===mx)ties++}hit[n]+=(cnt[0]===mx)?1/ties:0}}
      for(n=1;n<=NM;n++)hit[n]/=TR;cache[key]=hit;return hit}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,pad=40,p=+$("#rsP").value,K=+$("#rsK").value,m=curve(p,K),X=function(n){return pad+(n-1)/(NM-1)*(w-pad-12)},Y=function(v){return h-24-v*(h-40)},i;
      g.clearRect(0,0,w,h);g.font="11px sans-serif";g.strokeStyle=c.line;for(i=0;i<=1;i+=.25){g.beginPath();g.moveTo(pad,Y(i));g.lineTo(w-8,Y(i));g.stroke();lab(g,Math.round(i*100)+"%",pad-6,Y(i)+4,c.muted,"right")}
      [1,8,16,24,32].forEach(function(n){lab(g,n,X(n),h-8,c.muted,"center")});lab(g,"samples N →",w-12,12,c.muted,"right");
      g.setLineDash([4,4]);g.strokeStyle=c.muted;g.beginPath();g.moveTo(pad,Y(p));g.lineTo(w-8,Y(p));g.stroke();g.setLineDash([]);
      function line(f,col){g.strokeStyle=col;g.lineWidth=2.6;g.beginPath();for(var n=1;n<=NM;n++){var y=Y(f(n));n===1?g.moveTo(X(n),y):g.lineTo(X(n),y)}g.stroke()}
      line(function(n){return 1-Math.pow(1-p,n)},c.c1);line(function(n){return m[n]},c.c2);lab(g,"pass@N",X(20),Y(1-Math.pow(1-p,20))-8,c.c1);lab(g,"majority vote",X(20),Y(m[20])+16,c.c2);
      $("#rsPV").textContent=p.toFixed(2);$("#rsKV").textContent=K;$("#rsA").textContent=Math.round(p*100)+"%";$("#rsB").textContent=Math.round(m[16]*100)+"%";$("#rsC").textContent=Math.round((1-Math.pow(1-p,16))*100)+"%"});
    $("#rsP").addEventListener("input",function(){V.draw()});$("#rsK").addEventListener("input",function(){V.draw()})})();

  /* judge bias */
  (function(){
    var cc=$("#appEv");if(!cc)return;var seed=3;
    function lg(p){return Math.log(p/(1-p))}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,W=+$("#evW").value,B=+$("#evB").value,N=+$("#evN").value,r=rng(seed),a=0,b=0,i;
      var pA=sig(lg(W)+B),pB=1-sig(lg(1-W)+B);for(i=0;i<N;i++){if(r()<pA)a++;var x1=r()<pA,x2=r()<pB;b+=((x1?1:0)+(x2?1:0))/2}
      var vals=[[W,"true",c.muted],[a/N,"A always first",c.c2],[b/N,"both orders",c.c1]],bw=(w-60)/3,Y=function(v){return h-26-v*(h-44)};
      g.clearRect(0,0,w,h);g.font="11px sans-serif";g.strokeStyle=c.line;[0,.5,1].forEach(function(t){g.beginPath();g.moveTo(30,Y(t));g.lineTo(w-8,Y(t));g.stroke();lab(g,Math.round(t*100)+"%",26,Y(t)+4,c.muted,"right")});
      vals.forEach(function(v,k){var x=40+k*bw+bw*.2,ww=bw*.6;g.fillStyle=v[2];g.fillRect(x,Y(v[0]),ww,h-26-Y(v[0]));lab(g,v[1],x+ww/2,h-8,c.muted,"center");lab(g,Math.round(v[0]*100)+"%",x+ww/2,Y(v[0])-6,c.ink,"center");
        if(k>0){var se=1.96*Math.sqrt(v[0]*(1-v[0])/N),xm=x+ww/2;g.strokeStyle=c.ink;g.lineWidth=1.5;g.beginPath();g.moveTo(xm,Y(Math.min(1,v[0]+se)));g.lineTo(xm,Y(Math.max(0,v[0]-se)));g.stroke()}});
      $("#evWV").textContent=W.toFixed(2);$("#evBV").textContent=B.toFixed(2);$("#evNV").textContent=N;$("#evT").textContent=Math.round(W*100)+"%";$("#evX").textContent=Math.round(a/N*100)+"%";$("#evY").textContent=Math.round(b/N*100)+"%"});
    ["evW","evB","evN"].forEach(function(id){$("#"+id).addEventListener("input",function(){V.draw()})});$("#evGo").addEventListener("click",function(){seed=Math.floor(Math.random()*9999);V.draw()})})();

  /* CLIP */
  (function(){
    var cc=$("#appClip");if(!cc)return;var NN=5,IL=["dog","cat","car","tree","pizza"],run=true,th,ph,acc=0;
    function reset(){var r=rng(Math.floor(Math.random()*999)+1);th=[];ph=[];for(var i=0;i<NN;i++){th.push(r()*6.28);ph.push(r()*6.28)}}
    function loss(t,p,T){var L=0,i,j;for(i=0;i<NN;i++){var s=[],s2=[];for(j=0;j<NN;j++){s.push(Math.cos(t[i]-p[j])/T);s2.push(Math.cos(t[j]-p[i])/T)}
      var m=Math.max.apply(null,s),z=0,m2=Math.max.apply(null,s2),z2=0;s.forEach(function(v){z+=Math.exp(v-m)});s2.forEach(function(v){z2+=Math.exp(v-m2)});L+=(Math.log(z)+m-s[i]+Math.log(z2)+m2-s2[i])/2}return L/NN}
    function step(){var T=+$("#clT").value,e=1e-4,gt=[],gp=[],i;for(i=0;i<NN;i++){var a=th.slice(),b=th.slice();a[i]+=e;b[i]-=e;gt.push((loss(a,ph,T)-loss(b,ph,T))/(2*e));a=ph.slice();b=ph.slice();a[i]+=e;b[i]-=e;gp.push((loss(th,a,T)-loss(th,b,T))/(2*e))}
      for(i=0;i<NN;i++){th[i]-=.35*gt[i]*T*2;ph[i]-=.35*gp[i]*T*2}}
    reset();
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,T=+$("#clT").value,R=Math.min(h*.4,w*.2),cx=w*.22,cy=h/2,i,j;g.clearRect(0,0,w,h);g.font="11px sans-serif";
      g.strokeStyle=c.line;g.beginPath();g.arc(cx,cy,R,0,6.3);g.stroke();
      for(i=0;i<NN;i++){var x1=cx+R*Math.cos(th[i]),y1=cy+R*Math.sin(th[i]),x2=cx+R*Math.cos(ph[i]),y2=cy+R*Math.sin(ph[i]);g.strokeStyle=rgba(c.muted,.5);g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();
        g.fillStyle=c.c1;g.beginPath();g.arc(x1,y1,6,0,6.3);g.fill();g.fillStyle=c.c2;g.beginPath();g.arc(x2,y2,6,0,6.3);g.fill();lab(g,IL[i],x1+(Math.cos(th[i])>0?9:-9),y1+4,c.c1,Math.cos(th[i])>0?"left":"right")}
      var mx=w*.5,sz=Math.min(h-50,w*.3),cs=sz/NN,hit=0;lab(g,"similarity (rows: images, columns: captions)",mx,16,c.muted);
      for(i=0;i<NN;i++){var best=0,bv=-9;for(j=0;j<NN;j++){var s=Math.cos(th[i]-ph[j]);if(s>bv){bv=s;best=j}var t=(s+1)/2;g.fillStyle=rgba(c.c2,.1+.9*t*t);g.fillRect(mx+j*cs,26+i*cs,cs-2,cs-2);if(i===j){g.strokeStyle=c.c1;g.lineWidth=2;g.strokeRect(mx+j*cs,26+i*cs,cs-2,cs-2);g.lineWidth=1}}if(best===i)hit++;lab(g,IL[i],mx-6,26+i*cs+cs/2+4,c.muted,"right")}
      $("#clTV").textContent=T.toFixed(2);$("#clL").textContent=loss(th,ph,T).toFixed(3);$("#clA").textContent=hit+" of "+NN});
    var Lp=loopOn(cc,function(dt){acc+=dt;if(run&&acc>40){acc=0;step()}V.draw()});if(!rm)Lp.set(true);
    $("#clRun").addEventListener("click",function(){run=!run;this.textContent=run?"Pause":"Train"});$("#clRe").addEventListener("click",function(){reset();V.draw()});
    $("#clT").addEventListener("input",function(){V.draw()});reset();V.draw()})();
})();
