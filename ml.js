/* Classical ML demos */
(function(){
  function f(n,d){return isFinite(n)?n.toFixed(d):"∞"}
  function erf(x){var s=x<0?-1:1;x=Math.abs(x);var t=1/(1+0.3275911*x);return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x))}
  function Phi(z){return 0.5*(1+erf(z/Math.SQRT2))}

  /* picker */
  (function(){
    var d=$("#selData"),l=$("#selLab"),s=$("#selSize"),out=$("#pickOut");if(!d)return;
    function run(){
      var D=d.value,L=l.value,S=s.value,a=[];
      if(L==="none"){
        a=["Scale the features, then try k-means and look at the elbow and silhouette score.","Use PCA to see the shape of the data in two dimensions.","If clusters are not round, try DBSCAN or a Gaussian mixture. For outliers, try Isolation Forest."];
        if(D==="txt")a.unshift("Embed the text with a pretrained sentence-embedding model, then cluster the embeddings.");
        if(D==="img")a.unshift("Extract features with a pretrained vision model, then cluster those features.");
        if(D==="ts")a.unshift("Cut the series into windows and cluster summary features, or use change-point detection.");
      }else if(D==="tab"){
        a=[L==="num"?"Baseline: predict the mean, then Ridge regression on scaled features.":"Baseline: predict the majority class, then logistic regression on scaled features."];
        a.push(S==="l"?"Gradient boosting (LightGBM, XGBoost or HistGradientBoosting) is the usual strong choice at this size.":S==="m"?"Gradient boosting is the usual strong choice, with a random forest as a sanity check.":"Random forest and regularised linear models are quick and hard to overfit badly. Add boosting with early stopping.");
        a.push("Use cross-validation, group splits if rows share an entity, and permutation importance to explain results.");
        if(S==="l")a.push("A neural network on tabular data rarely beats boosting. Try it only if you have embeddings or very high-cardinality categories.");
      }else if(D==="img"){
        a=["Start with transfer learning: fine-tune a pretrained CNN or vision transformer.",S==="s"?"With little data, freeze most layers and use heavy augmentation.":"With lots of data, fine-tune the whole network and tune the learning rate carefully.","Baseline first: a linear classifier on frozen pretrained features is surprisingly strong."];
      }else if(D==="txt"){
        a=["Baseline: TF-IDF features with logistic regression. It trains in seconds.",S==="s"?"Next, sentence embeddings plus a linear model, or fine-tune a small pretrained transformer.":"Next, fine-tune a pretrained transformer, or use an LLM with a good prompt and evaluate it on a labelled sample.","Check for duplicate or near-duplicate texts between train and test."];
      }else{
        a=["Baseline: predict the last value, or the value from the same time last season.","Next, gradient boosting on lag and calendar features, or ARIMA and exponential smoothing for single series.","Always split by time and validate on later data than you trained on."];
      }
      out.innerHTML="<h4>Start here</h4><ul>"+a.map(function(x){return "<li>"+esc(x)+"</li>"}).join("")+"</ul>"}
    [d,l,s].forEach(function(e){e.addEventListener("change",run)});run()})();

  /* 1. gradient descent */
  (function(){
    var cf=$("#gdFit");if(!cf)return;
    var r=rng(7),X=[],Y=[],i;for(i=0;i<25;i++){var x=0.05+0.9*r();X.push(x);Y.push(0.6*x+0.25+gauss(r)*0.05)}
    var W=-0.5,B=0.95,hist=[1],run=false,acc=0,div=false;
    function mse(w,b){var s=0;for(var i=0;i<X.length;i++){var e=w*X[i]+b-Y[i];s+=e*e}return s/X.length}
    function reset(){W=-0.5;B=0.95;hist=[mse(W,B)];div=false;run=false;L.set(false);$("#gdRun").textContent="Run";draw()}
    function step(){
      var lr=+$("#gdLr").value,dw=0,db=0,n=X.length;
      for(var i=0;i<n;i++){var e=W*X[i]+B-Y[i];dw+=2*e*X[i]/n;db+=2*e/n}
      W-=lr*dw;B-=lr*db;var l=mse(W,B);hist.push(l);
      if(!isFinite(l)||l>1e8){div=true;run=false;L.set(false);$("#gdRun").textContent="Run"}}
    var fit=cv(cf,function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);axes(g,w,h,c);
      var sx=function(x){return x*w},sy=function(y){return (1-y)*h};
      g.strokeStyle=rgba(c.c2,.35);g.lineWidth=1;
      X.forEach(function(x,i){var p=W*x+B;g.beginPath();g.moveTo(sx(x),sy(Y[i]));g.lineTo(sx(x),sy(clamp(p,-.2,1.2)));g.stroke()});
      X.forEach(function(x,i){dot(g,sx(x),sy(Y[i]),4,c.c1)});
      g.strokeStyle=c.accent;g.lineWidth=2.5;g.beginPath();g.moveTo(0,sy(clamp(B,-5,5)));g.lineTo(w,sy(clamp(W+B,-5,5)));g.stroke()});
    var ls=cv($("#gdLoss"),function(o,c){var g=o.g,w=o.w,h=o.h,pad=34;g.clearRect(0,0,w,h);
      var lo=-5,hi=1,n=Math.max(60,hist.length-1);
      g.font="11px "+getComputedStyle(document.body).fontFamily;g.fillStyle=c.muted;g.strokeStyle=c.line;g.lineWidth=1;
      for(var e=lo;e<=hi;e+=2){var y=h-14-(e-lo)/(hi-lo)*(h-24);g.beginPath();g.moveTo(pad,y);g.lineTo(w,y);g.stroke();g.fillText("1e"+e,2,y+4)}
      g.strokeStyle=c.c2;g.lineWidth=2;g.beginPath();
      hist.forEach(function(v,i){var e=clamp(Math.log10(Math.max(v,1e-12)),lo,hi);var x=pad+i/n*(w-pad-6),y=h-14-(e-lo)/(hi-lo)*(h-24);if(i)g.lineTo(x,y);else g.moveTo(x,y)});g.stroke();
      g.fillStyle=c.muted;g.fillText("steps →",w-52,h-2);
      if(div){g.fillStyle=c.c2;g.font="600 13px "+getComputedStyle(document.body).fontFamily;g.fillText("Diverged: learning rate too high",pad+8,22)}});
    function draw(){fit.draw();ls.draw();$("#gdN").textContent=hist.length-1;$("#gdW").textContent=f(W,3);$("#gdB").textContent=f(B,3);$("#gdL").textContent=div?"diverged":f(hist[hist.length-1],4)}
    var L=loopOn(cf,function(dt){acc+=dt/45;while(acc>=1&&run){step();acc--}draw()});
    $("#gdRun").addEventListener("click",function(){if(div)reset();run=!run;this.textContent=run?"Pause":"Run";L.set(run)});
    $("#gdStep").addEventListener("click",function(){if(div)reset();step();draw()});
    $("#gdReset").addEventListener("click",reset);
    $("#gdLr").addEventListener("input",function(){$("#gdLrV").textContent=(+this.value).toFixed(2)});
    reset()})();

  /* 2. threshold */
  (function(){
    var cd=$("#thDist");if(!cd)return;
    var sig=0.15;
    function stats(){var t=+$("#thT").value,pi=$("#thP").value/100,sep=+$("#thS").value,mn=0.5-sep*sig/2,mp=0.5+sep*sig/2,N=1000,pos=N*pi,neg=N-pos;
      var tpr=1-Phi((t-mp)/sig),fpr=1-Phi((t-mn)/sig),TP=pos*tpr,FN=pos-TP,FP=neg*fpr,TN=neg-FP;
      return {t:t,pi:pi,mn:mn,mp:mp,tpr:tpr,fpr:fpr,TP:TP,FP:FP,FN:FN,TN:TN,sep:sep}}
    function pdf(x,m){return Math.exp(-0.5*Math.pow((x-m)/sig,2))/(sig*2.5066)}
    var d1=cv(cd,function(o,c){var s=stats(),g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);axes(g,w,h,c);
      var mx=0,x,i;for(i=0;i<=100;i++){x=i/100;mx=Math.max(mx,(1-s.pi)*pdf(x,s.mn),s.pi*pdf(x,s.mp))}
      function area(m,wt,col){g.beginPath();g.moveTo(0,h-1);for(var i=0;i<=100;i++){var x=i/100;g.lineTo(x*w,h-4-(wt*pdf(x,m)/mx)*(h-20))}g.lineTo(w,h-1);g.closePath();g.fillStyle=rgba(col,.35);g.fill();g.strokeStyle=col;g.lineWidth=2;g.stroke()}
      area(s.mn,1-s.pi,c.c1);area(s.mp,s.pi,c.c2);
      g.strokeStyle=c.ink;g.lineWidth=2;g.setLineDash([5,4]);g.beginPath();g.moveTo(s.t*w,4);g.lineTo(s.t*w,h);g.stroke();g.setLineDash([]);
      g.fillStyle=c.muted;g.font="11px sans-serif";g.fillText("score →",w-48,h-6);g.fillText("flag as positive →",Math.min(s.t*w+6,w-110),14)});
    var d2=cv($("#thRoc"),function(o,c){var s=stats(),g=o.g,w=o.w,h=o.h,pad=26;g.clearRect(0,0,w,h);
      var X=function(v){return pad+v*(w-pad-8)},Y=function(v){return h-pad-v*(h-pad-8)};
      g.strokeStyle=c.line;g.lineWidth=1;g.strokeRect(X(0),Y(1),X(1)-X(0),Y(0)-Y(1));g.setLineDash([4,4]);g.beginPath();g.moveTo(X(0),Y(0));g.lineTo(X(1),Y(1));g.stroke();g.setLineDash([]);
      g.strokeStyle=c.accent;g.lineWidth=2.5;g.beginPath();var auc=0,px=1,py=1;
      for(var t=-0.3;t<=1.3;t+=0.01){var tp=1-Phi((t-s.mp)/sig),fp=1-Phi((t-s.mn)/sig);if(t<-0.29)g.moveTo(X(fp),Y(tp));else g.lineTo(X(fp),Y(tp));auc+=(px-fp)*(py+tp)/2;px=fp;py=tp}g.stroke();
      dot(g,X(s.fpr),Y(s.tpr),6,c.c2,c.bg);
      g.fillStyle=c.muted;g.font="11px sans-serif";g.fillText("false positive rate →",X(0.3),h-8);g.save();g.translate(10,Y(0.2));g.rotate(-Math.PI/2);g.fillText("recall",0,0);g.restore();
      g.fillStyle=c.ink;g.fillText("AUC "+auc.toFixed(2),X(0.55),Y(0.18))});
    function upd(){var s=stats();$("#thTV").textContent=s.t.toFixed(2);$("#thPV").textContent=Math.round(s.pi*100)+"%";$("#thSV").textContent=s.sep.toFixed(1);
      var pr=s.TP+s.FP>0?s.TP/(s.TP+s.FP):NaN,re=s.TP/(s.TP+s.FN),f1=2*pr*re/(pr+re),ac=(s.TP+s.TN)/1000;
      $("#thPr").textContent=isNaN(pr)?"-":pr.toFixed(2);$("#thRe").textContent=re.toFixed(2);$("#thF").textContent=isNaN(f1)?"-":f1.toFixed(2);$("#thAc").textContent=ac.toFixed(3);
      $("#thTP").textContent=Math.round(s.TP);$("#thFP").textContent=Math.round(s.FP);$("#thFN").textContent=Math.round(s.FN);$("#thTN").textContent=Math.round(s.TN);d1.draw();d2.draw()}
    ["thT","thP","thS"].forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* 3. kNN */
  (function(){
    var cc=$("#knnCv");if(!cc)return;
    var pts=[],cur=0;
    function gen(){pts=blobs($("#knnD").value,80,+$("#knnN").value,3)}
    function predict(x,y,k,skip){var d=[],i,p;for(i=0;i<pts.length;i++){if(i===skip)continue;p=pts[i];d.push([(p.x-x)*(p.x-x)+(p.y-y)*(p.y-y),p.c])}
      d.sort(function(a,b){return a[0]-b[0]});var s=0,kk=Math.min(k,d.length);for(i=0;i<kk;i++)s+=d[i][1];return s/kk}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,k=+$("#knnK").value;g.clearRect(0,0,w,h);
      heatmap(g,w,h,function(x,y){var p=predict(x,y,k,-1);return p>.5?1:p<.5?0:.5},c.c1,c.c2,9,.28);
      pts.forEach(function(p){dot(g,p.x*w,(1-p.y)*h,5,p.c?c.c2:c.c1,c.bg)})});
    function upd(){var k=+$("#knnK").value,tr=0,lo=0;$("#knnKV").textContent=k;$("#knnNV").textContent=(+$("#knnN").value).toFixed(2);
      pts.forEach(function(p,i){if((predict(p.x,p.y,k,-1)>.5?1:0)===p.c)tr++;if((predict(p.x,p.y,k,i)>.5?1:0)===p.c)lo++});
      $("#knnCnt").textContent=pts.length;$("#knnTr").textContent=(100*tr/pts.length).toFixed(0)+"%";$("#knnLoo").textContent=(100*lo/pts.length).toFixed(0)+"%";V.draw()}
    cc.addEventListener("pointerdown",function(e){var m=mouse(cc,e),r=cc.getBoundingClientRect();pts.push({x:m.x/r.width,y:1-m.y/r.height,c:cur});upd()});
    $("#knnA").addEventListener("click",function(){cur=0;this.setAttribute("aria-pressed","true");$("#knnB").setAttribute("aria-pressed","false")});
    $("#knnB").addEventListener("click",function(){cur=1;this.setAttribute("aria-pressed","true");$("#knnA").setAttribute("aria-pressed","false")});
    $("#knnK").addEventListener("input",upd);
    ["knnD","knnN"].forEach(function(id){$("#"+id).addEventListener("input",function(){gen();upd()})});
    gen();upd()})();

  /* 4. decision tree */
  (function(){
    var cc=$("#trCv");if(!cc)return;
    var seed=11,pts,tr,te,tree;
    function gen(){pts=blobs($("#trK").value,140,+$("#trN").value,seed);tr=[];te=[];pts.forEach(function(p,i){(i%10<7?tr:te).push(p)});tree=null}
    function gini(a,n){return n?1-(a*a+(n-a)*(n-a))/(n*n):0}
    function grow(S,d,maxD){
      var n=S.length,ones=0;S.forEach(function(p){ones+=p.c});
      if(d>=maxD||ones===0||ones===n||n<2)return {p:n?ones/n:0.5};
      var best=null,ax,i;
      ["x","y"].forEach(function(ax){var A=S.slice().sort(function(a,b){return a[ax]-b[ax]}),L=0;
        for(i=0;i<n-1;i++){L+=A[i].c;if(A[i].x===A[i+1].x&&ax==="x"||A[i][ax]===A[i+1][ax])continue;
          var nl=i+1,nr=n-nl,sc=(nl*gini(L,nl)+nr*gini(ones-L,nr))/n;
          if(!best||sc<best.sc)best={sc:sc,ax:ax,t:(A[i][ax]+A[i+1][ax])/2}}});
      if(!best)return {p:ones/n};
      var l=S.filter(function(p){return p[best.ax]<=best.t}),r=S.filter(function(p){return p[best.ax]>best.t});
      if(!l.length||!r.length)return {p:ones/n};
      return {ax:best.ax,t:best.t,l:grow(l,d+1,maxD),r:grow(r,d+1,maxD)}}
    function pr(nd,x,y){while(nd.l)nd=((nd.ax==="x"?x:y)<=nd.t)?nd.l:nd.r;return nd.p}
    function leaves(nd){return nd.l?leaves(nd.l)+leaves(nd.r):1}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!tree)return;
      heatmap(g,w,h,function(x,y){return pr(tree,x,y)},c.c1,c.c2,6,.3);
      tr.forEach(function(p){dot(g,p.x*w,(1-p.y)*h,4.5,p.c?c.c2:c.c1,c.bg)});
      te.forEach(function(p){dot(g,p.x*w,(1-p.y)*h,5,null,p.c?c.c2:c.c1)})});
    function acc(S){var k=0;S.forEach(function(p){if((pr(tree,p.x,p.y)>.5?1:0)===p.c)k++});return 100*k/S.length}
    function upd(){var d=+$("#trD").value;$("#trDV").textContent=d;$("#trNV").textContent=(+$("#trN").value).toFixed(2);tree=grow(tr,0,d);
      $("#trLv").textContent=leaves(tree);$("#trTr").textContent=acc(tr).toFixed(0)+"%";$("#trTe").textContent=acc(te).toFixed(0)+"%";V.draw()}
    ["trD"].forEach(function(id){$("#"+id).addEventListener("input",upd)});
    ["trN","trK"].forEach(function(id){$("#"+id).addEventListener("input",function(){gen();upd()})});
    $("#trNew").addEventListener("click",function(){seed=Math.floor(Math.random()*9999);gen();upd()});
    gen();upd()})();

  /* 5. k-means */
  (function(){
    var cc=$("#kmCv");if(!cc)return;
    var P,K,cen,asg,phase,it,seed=5;
    function gen(){var r=rng(seed),nc=3+Math.floor(r()*3),cs=[],i;for(i=0;i<nc;i++)cs.push([0.15+0.7*r(),0.15+0.7*r()]);
      P=[];for(i=0;i<130;i++){var c=cs[i%nc];P.push([clamp(c[0]+gauss(r)*0.065,0.02,0.98),clamp(c[1]+gauss(r)*0.065,0.02,0.98)])}}
    function d2(a,b){return (a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1])}
    function assign(cs){var ch=false,a=P.map(function(p,i){var b=0,bd=1e9;cs.forEach(function(c,j){var d=d2(p,c);if(d<bd){bd=d;b=j}});return b});return a}
    function update(a,cs){return cs.map(function(c,j){var sx=0,sy=0,n=0;P.forEach(function(p,i){if(a[i]===j){sx+=p[0];sy+=p[1];n++}});return n?[sx/n,sy/n]:c})}
    function inertia(a,cs){var s=0;P.forEach(function(p,i){s+=d2(p,cs[a[i]])});return s}
    function init(){K=+$("#kmK").value;var r=rng(Math.floor(Math.random()*1e6)),idx=[];while(idx.length<Math.min(K,P.length)){var i=Math.floor(r()*P.length);if(idx.indexOf(i)<0)idx.push(i)}
      cen=idx.map(function(i){return P[i].slice()});asg=null;phase="assign";it=0}
    function step(){if(phase==="assign"){var a=assign(cen),same=asg&&a.every(function(v,i){return v===asg[i]});asg=a;phase="update";return !!same}
      var n=update(asg,cen),moved=n.some(function(c,j){return Math.abs(c[0]-cen[j][0])+Math.abs(c[1]-cen[j][1])>1e-6});cen=n;phase="assign";it++;return !moved}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,pal=[c.c1,c.c2,c.c3,c.c4,c.warn,c.accent,c.ink,c.muted];g.clearRect(0,0,w,h);if(!P||!cen)return;
      P.forEach(function(p,i){dot(g,p[0]*w,(1-p[1])*h,4.5,asg?pal[asg[i]%8]:c.muted)});
      cen.forEach(function(q,j){dot(g,q[0]*w,(1-q[1])*h,10,pal[j%8],c.ink)})});
    function ke(Kk){var best=1e9,rr=rng(Kk*77);for(var t=0;t<6;t++){var cs=[],i,used={};while(cs.length<Kk){i=Math.floor(rr()*P.length);if(!used[i]){used[i]=1;cs.push(P[i].slice())}}
      var a=null,l;for(l=0;l<40;l++){var na=assign(cs);if(a&&na.every(function(v,i){return v===a[i]}))break;a=na;cs=update(a,cs)}best=Math.min(best,inertia(a,cs))}return best}
    var el=null,elb=cv($("#kmElbow"),function(o,c){var g=o.g,w=o.w,h=o.h,pad=34;g.clearRect(0,0,w,h);if(!el)return;var mx=el[0],X=function(k){return pad+(k-1)/7*(w-pad-14)},Y=function(v){return h-26-v/mx*(h-44)};
      g.strokeStyle=c.line;g.lineWidth=1;g.beginPath();g.moveTo(pad,8);g.lineTo(pad,h-26);g.lineTo(w-8,h-26);g.stroke();
      g.strokeStyle=c.accent;g.lineWidth=2.5;g.beginPath();el.forEach(function(v,i){i?g.lineTo(X(i+1),Y(v)):g.moveTo(X(1),Y(v))});g.stroke();
      g.fillStyle=c.muted;g.font="11px sans-serif";el.forEach(function(v,i){dot(g,X(i+1),Y(v),i+1===K?7:3.5,i+1===K?c.c2:c.accent);g.fillText(i+1,X(i+1)-3,h-10)});g.fillText("inertia",4,14);g.fillText("k",w-14,h-10)});
    function upd(){$("#kmKV").textContent=K;$("#kmPh").textContent=phase;$("#kmIt").textContent=it;$("#kmIn").textContent=asg?inertia(asg,cen).toFixed(2):"-";V.draw();elb.draw()}
    function newdata(){gen();el=[];for(var k=1;k<=8;k++)el.push(ke(k));init();upd()}
    $("#kmStep").addEventListener("click",function(){step();upd()});
    $("#kmRun").addEventListener("click",function(){var n=0;while(n++<60&&!step()){}if(phase==="update")step();upd()});
    $("#kmInit").addEventListener("click",function(){init();upd()});
    $("#kmNew").addEventListener("click",function(){seed=Math.floor(Math.random()*9999);newdata()});
    $("#kmK").addEventListener("input",function(){init();upd()});
    newdata()})();

  /* 6. bias-variance */
  (function(){
    var cf=$("#bvFit");if(!cf)return;
    var LAM=[0,1e-6,1e-4,1e-3,1e-2,1e-1,1],seed=2,TR,TE;
    function truth(x){return Math.sin(2*Math.PI*x)}
    function gen(){var r=rng(seed),i;TR=[];for(i=0;i<20;i++){var x=r();TR.push([x,truth(x)+gauss(r)*0.3])}
      var r2=rng(999);TE=[];for(i=0;i<200;i++){var x2=i/199;TE.push([x2,truth(x2)+gauss(r2)*0.3])}}
    function feat(x,d){var t=2*x-1,a=[],k;for(k=0;k<=d;k++)a.push(Math.pow(t,k));return a}
    function solve(A,b){var n=b.length,i,j,k;for(i=0;i<n;i++){var p=i;for(j=i+1;j<n;j++)if(Math.abs(A[j][i])>Math.abs(A[p][i]))p=j;var t=A[i];A[i]=A[p];A[p]=t;var tb=b[i];b[i]=b[p];b[p]=tb;
      for(j=i+1;j<n;j++){var m=A[j][i]/(A[i][i]||1e-12);for(k=i;k<n;k++)A[j][k]-=m*A[i][k];b[j]-=m*b[i]}}
      var x=new Array(n);for(i=n-1;i>=0;i--){var s=b[i];for(j=i+1;j<n;j++)s-=A[i][j]*x[j];x[i]=s/(A[i][i]||1e-12)}return x}
    function fit(d,lam){var n=d+1,A=[],b=[],i,j;for(i=0;i<n;i++){A.push(new Array(n).fill(0));b.push(0)}
      TR.forEach(function(p){var f=feat(p[0],d);for(i=0;i<n;i++){b[i]+=f[i]*p[1];for(j=0;j<n;j++)A[i][j]+=f[i]*f[j]}});
      for(i=0;i<n;i++)A[i][i]+=lam+1e-10;return solve(A,b)}
    function pred(w,x){var f=feat(x,w.length-1),s=0;for(var i=0;i<f.length;i++)s+=w[i]*f[i];return s}
    function mse(w,S){var s=0;S.forEach(function(p){var e=pred(w,p[0])-p[1];s+=e*e});return s/S.length}
    var curve=[];
    var V=cv(cf,function(o,c){if(!TR)return;var g=o.g,w=o.w,h=o.h,d=+$("#bvD").value,lam=LAM[+$("#bvL").value],wt=fit(d,lam),sy=function(y){return h/2-y/3.4*h};g.clearRect(0,0,w,h);
      g.save();g.beginPath();g.rect(0,0,w,h);g.clip();
      g.strokeStyle=c.muted;g.lineWidth=1.5;g.setLineDash([5,4]);g.beginPath();for(var i=0;i<=100;i++){var x=i/100;i?g.lineTo(x*w,sy(truth(x))):g.moveTo(0,sy(truth(x)))}g.stroke();g.setLineDash([]);
      TR.forEach(function(p){dot(g,p[0]*w,sy(p[1]),4.5,c.c1)});
      g.strokeStyle=c.accent;g.lineWidth=2.6;g.beginPath();for(i=0;i<=200;i++){var x2=i/200,y=pred(wt,x2);i?g.lineTo(x2*w,sy(y)):g.moveTo(0,sy(y))}g.stroke();g.restore()});
    var E=cv($("#bvErr"),function(o,c){if(!TR||!curve.length)return;var g=o.g,w=o.w,h=o.h,pad=34,lam=LAM[+$("#bvL").value],d=+$("#bvD").value;g.clearRect(0,0,w,h);
      var lo=-2,hi=2,X=function(k){return pad+k/12*(w-pad-12)},Y=function(v){var e=clamp(Math.log10(Math.max(v,1e-9)),lo,hi);return h-24-(e-lo)/(hi-lo)*(h-40)};
      g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="11px sans-serif";for(var e=lo;e<=hi;e++){var y=Y(Math.pow(10,e));g.beginPath();g.moveTo(pad,y);g.lineTo(w-6,y);g.stroke();g.fillText("1e"+e,2,y+4)}
      [[0,c.c1],[1,c.c2]].forEach(function(s){g.strokeStyle=s[1];g.lineWidth=2.4;g.beginPath();curve.forEach(function(r,k){var x=X(k),y2=Y(r[s[0]]);k?g.lineTo(x,y2):g.moveTo(x,y2)});g.stroke();curve.forEach(function(r,k){dot(g,X(k),Y(r[s[0]]),k===d?6:2.5,s[1])})});
      g.fillStyle=c.muted;g.fillText("polynomial degree →",w-112,h-6);g.strokeStyle=c.ink;g.setLineDash([3,3]);g.beginPath();g.moveTo(X(d),6);g.lineTo(X(d),h-24);g.stroke();g.setLineDash([])});
    function upd(){var d=+$("#bvD").value,lam=LAM[+$("#bvL").value];$("#bvDV").textContent=d;$("#bvLV").textContent=lam===0?"0":lam.toExponential(0).replace("e-","e−");
      curve=[];for(var k=0;k<=12;k++){var w=fit(k,lam);curve.push([mse(w,TR),mse(w,TE)])}
      var tr=curve[d][0],te=curve[d][1];$("#bvTr").textContent=tr.toFixed(3);$("#bvTe").textContent=te>100?te.toExponential(1):te.toFixed(3);
      $("#bvV").textContent=tr>0.16?"underfit":(te>tr*2.2&&te>0.14?"overfit":"good fit");V.draw();E.draw()}
    ["bvD","bvL"].forEach(function(id){$("#"+id).addEventListener("input",upd)});
    $("#bvNew").addEventListener("click",function(){seed=Math.floor(Math.random()*9999);gen();upd()});
    gen();upd()})();
})();
autoRun('gdRun');
autoRun('kmRun');
/* permutation importance */
(function(){
  var cc=$("#exCv");if(!cc)return;var seed=3;function lab(g,t,x,y,col,al){g.fillStyle=col;g.textAlign=al||"left";g.fillText(t,x,y)}
  function solve(A,b){var n=b.length,i,j,k;A=A.map(function(r){return r.slice()});b=b.slice();for(i=0;i<n;i++){var m=i;for(j=i+1;j<n;j++)if(Math.abs(A[j][i])>Math.abs(A[m][i]))m=j;var t=A[i];A[i]=A[m];A[m]=t;var tb=b[i];b[i]=b[m];b[m]=tb;
    for(j=i+1;j<n;j++){var f=A[j][i]/A[i][i];for(k=i;k<n;k++)A[j][k]-=f*A[i][k];b[j]-=f*b[i]}}
    var x=new Array(n).fill(0);for(i=n-1;i>=0;i--){var s=b[i];for(j=i+1;j<n;j++)s-=A[i][j]*x[j];x[i]=s/A[i][i]}return x}
  function data(n,rho,r){var X=[],y=[];for(var i=0;i<n;i++){var f1=gauss(r),f3=rho*f1+Math.sqrt(1-rho*rho)*gauss(r),f2=gauss(r),f4=gauss(r);X.push([f1,f2,f3,f4]);y.push(3*f1+1.5*f2+gauss(r)*.5)}return [X,y]}
  function mse(X,y,w){var s=0;for(var i=0;i<X.length;i++){var p=0;for(var j=0;j<4;j++)p+=w[j]*X[i][j];s+=(p-y[i])*(p-y[i])}return s/X.length}
  var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,rho=+$("#exC").value,lam=+$("#exL").value,r=rng(seed),T=data(250,rho,r),H=data(300,rho,r),A=[],b=[],i,j,k;
    for(i=0;i<4;i++){A.push([]);var s=0;for(j=0;j<4;j++){var v=0;for(k=0;k<250;k++)v+=T[0][k][i]*T[0][k][j];A[i].push(v+(i===j?lam:0))}for(k=0;k<250;k++)s+=T[0][k][i]*T[1][k];b.push(s)}
    var W=solve(A,b),base=mse(H[0],H[1],W),imp=[];
    for(j=0;j<4;j++){var tot=0;for(var rep=0;rep<6;rep++){var Xp=H[0].map(function(row){return row.slice()}),perm=[],q;for(q=0;q<300;q++)perm.push(q);for(q=299;q>0;q--){var z=Math.floor(r()*(q+1)),tmp=perm[q];perm[q]=perm[z];perm[z]=tmp}for(q=0;q<300;q++)Xp[q][j]=H[0][perm[q]][j];tot+=mse(Xp,H[1],W)-base}imp.push(tot/6)}
    var mx=Math.max.apply(null,imp)*1.1||1,bw=(w-120)/4,cols=[c.c1,c.c1,c.c2,c.muted],names=["feature 1","feature 2","feature 3 (copy of 1)","feature 4 (noise)"];
    g.clearRect(0,0,w,h);g.font="11px sans-serif";for(i=0;i<4;i++){var bh=Math.max(1,imp[i]/mx*(h-50));g.fillStyle=cols[i];g.fillRect(60+i*bw+bw*.2,h-28-bh,bw*.6,bh);lab(g,names[i],60+i*bw+bw*.5,h-10,c.muted,"center");lab(g,imp[i].toFixed(1),60+i*bw+bw*.5,h-32-bh,c.ink,"center")}
    lab(g,"increase in error",6,14,c.muted);$("#exCV").textContent=rho.toFixed(2);$("#exLV").textContent=lam;$("#exW").textContent=W.map(function(x){return x.toFixed(2)}).join(", ");$("#exE").textContent=base.toFixed(3)});
  $("#exC").addEventListener("input",function(){V.draw()});$("#exL").addEventListener("input",function(){V.draw()});$("#exNew").addEventListener("click",function(){seed=Math.floor(Math.random()*9999);V.draw()})})();
