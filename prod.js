/* Production and scaling demos */
(function(){
  function lab(g,t,x,y,col,al){g.fillStyle=col;g.textAlign=al||"left";g.fillText(t,x,y)}
  function erf(x){var s=x<0?-1:1;x=Math.abs(x);var t=1/(1+.3275911*x);return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x))}
  function Phi(z){return .5*(1+erf(z/Math.SQRT2))}
  function colorsOf(c){return [c.c1,c.c2,c.c3,c.c4]}

  /* sharding */
  (function(){
    var cc=$("#prPar");if(!cc)return;
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,P=+$("#pmP").value,n=+$("#pmN").value,S=[["DDP",2*P,2*P,12*P],["ZeRO-1",2*P,2*P,12*P/n],["ZeRO-2",2*P,2*P/n,12*P/n],["ZeRO-3 / FSDP",2*P/n,2*P/n,12*P/n]],mx=0,i;
      S.forEach(function(s){mx=Math.max(mx,s[1]+s[2]+s[3])});mx=Math.max(100,mx*1.05);var lx=118,X=function(v){return lx+v/mx*(w-lx-70)},rh=(h-30)/4;
      g.clearRect(0,0,w,h);g.font="11px sans-serif";
      S.forEach(function(s,k){var y=14+k*rh,cur=0,tot=s[1]+s[2]+s[3];lab(g,s[0],lx-8,y+rh*.4,c.muted,"right");[c.c1,c.c3,c.c2].forEach(function(col,j){g.fillStyle=col;g.fillRect(X(cur),y,Math.max(0,X(cur+s[j+1])-X(cur)-1),rh*.55);cur+=s[j+1]});
        lab(g,tot.toFixed(1)+" GB",X(tot)+6,y+rh*.4,tot>80?c.c2:c.ink);$("#pm"+k).textContent=tot.toFixed(1)+" GB "+(tot<=80?"(fits 80 GB)":"(too big)")});
      [[40,"40 GB"],[80,"80 GB"]].forEach(function(l){g.strokeStyle=c.muted;g.setLineDash([4,4]);g.beginPath();g.moveTo(X(l[0]),4);g.lineTo(X(l[0]),h-18);g.stroke();g.setLineDash([]);lab(g,l[1],X(l[0]),h-5,c.muted,"center")});
      lab(g,"weights",lx,h-5,c.c1);lab(g,"gradients",lx+60,h-5,c.c3);lab(g,"optimiser state",lx+130,h-5,c.c2);
      $("#pmPV").textContent=P;$("#pmNV").textContent=n});
    ["pmP","pmN"].forEach(function(id){$("#"+id).addEventListener("input",function(){V.draw()})})})();

  /* pipeline */
  (function(){
    var cc=$("#prPipe");if(!cc)return;var t=0;
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,S=+$("#plS").value,M=+$("#plM").value,F=M+S-1,T=2*F,pad=44,cw=(w-pad-8)/T,ch=(h-24)/S,i,s,m,cur=rm?T:(t/260)%(T+6);
      g.clearRect(0,0,w,h);g.font="10px sans-serif";
      for(s=0;s<S;s++){lab(g,"GPU "+s,pad-6,8+s*ch+ch/2+3,c.muted,"right");for(i=0;i<T;i++){g.fillStyle=rgba(c.muted,i<cur?.16:.06);g.fillRect(pad+i*cw+1,8+s*ch+1,cw-2,ch-2)}
        for(m=0;m<M;m++){var tf=s+m,tb=F+(S-1-s)+m;[[tf,c.c1],[tb,c.c2]].forEach(function(p){if(p[0]<cur){g.fillStyle=p[1];g.fillRect(pad+p[0]*cw+1,8+s*ch+1,cw-2,ch-2);if(cw>14){g.fillStyle=c.bg;g.textAlign="center";g.fillText(m+1,pad+p[0]*cw+cw/2,8+s*ch+ch/2+3)}}})}}
      g.strokeStyle=c.ink;g.beginPath();g.moveTo(pad+cur*cw,4);g.lineTo(pad+cur*cw,h-16);g.stroke();lab(g,"time →",w-8,h-4,c.muted,"right");
      var b=(S-1)/F;$("#plSV").textContent=S;$("#plMV").textContent=M;$("#plB").textContent=(b*100).toFixed(1)+"%";$("#plU").textContent=((1-b)*100).toFixed(1)+"%";$("#plT").textContent=T});
    ["plS","plM"].forEach(function(id){$("#"+id).addEventListener("input",function(){t=0;V.draw()})});
    var L=loopOn(cc,function(dt){t+=dt;V.draw()});if(!rm)L.set(true)})();

  /* batching */
  (function(){
    var cc=$("#prBatch");if(!cc)return;var N=14,len=[],t=0;
    function gen(){var r=rng(Math.floor(Math.random()*9999)+1);len=[];for(var i=0;i<N;i++)len.push(5+Math.floor(r()*r()*40)+Math.floor(r()*10))}gen();
    function sim(B){var st=[],ct=[],i,k,tm=0;for(k=0;k<B;k++){st.push([]);ct.push([])}
      for(i=0;i<N;i+=B){var grp=len.slice(i,i+B),mx=Math.max.apply(null,grp);grp.forEach(function(l,j){st[j].push({i:i+j,s:tm,l:l,end:tm+mx})});tm+=mx}var sT=tm,free=[];for(k=0;k<B;k++)free.push(0);
      for(i=0;i<N;i++){var m=0;for(k=1;k<B;k++)if(free[k]<free[m])m=k;ct[m].push({i:i,s:free[m],l:len[i],end:free[m]+len[i]});free[m]+=len[i]}
      return {st:st,ct:ct,sT:sT,cT:Math.max.apply(null,free)}}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,B=+$("#cbS").value,R=sim(B),mx=Math.max(R.sT,R.cT),pad=44,X=function(v){return pad+v/mx*(w-pad-8)},cur=rm?mx:(t/50)%(mx+10),col=colorsOf(c),ph=(h-16)/2;
      g.clearRect(0,0,w,h);g.font="11px sans-serif";
      [["Static",R.st,0,true],["Continuous",R.ct,1,false]].forEach(function(P){var y0=8+P[2]*ph,lh=(ph-18)/B;lab(g,P[0],pad-6,y0+ph/2-6,c.muted,"right");
        P[1].forEach(function(lane,k){lane.forEach(function(q){var x0=X(q.s),e=Math.min(cur,q.s+q.l);if(P[3]&&cur>q.s){g.fillStyle=rgba(c.muted,.12);g.fillRect(x0,y0+k*lh+1,X(Math.min(cur,q.end))-x0,lh-2)}
          if(e>q.s){g.fillStyle=col[q.i%4];g.globalAlpha=.9;g.fillRect(x0,y0+k*lh+1,X(e)-x0-1,lh-2);g.globalAlpha=1}})});});
      g.strokeStyle=c.ink;g.beginPath();g.moveTo(X(cur),4);g.lineTo(X(cur),h-14);g.stroke();lab(g,"decode steps →",w-8,h-3,c.muted,"right");
      $("#cbSV").textContent=B;$("#cbA").textContent=R.sT;$("#cbB").textContent=R.cT;$("#cbC").textContent=(R.sT/R.cT).toFixed(2)+"×"});
    $("#cbS").addEventListener("input",function(){t=0;V.draw()});$("#cbNew").addEventListener("click",function(){gen();t=0;V.draw()});
    var L=loopOn(cc,function(dt){t+=dt;V.draw()});if(!rm)L.set(true)})();

  /* KV cache */
  (function(){
    var cc=$("#prKv");if(!cc)return;var M={"8":[32,8,128,8],"70":[80,8,128,70],"7":[32,32,128,7]};
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,m=M[$("#kvM").value],T=+$("#kvT").value,B=+$("#kvB").value,d=+$("#kvD").value,per=2*m[0]*m[1]*m[2]*d,kv=per*T*B/1e9,wt=m[3]*2,tot=kv+wt,mx=Math.max(160,tot*1.05),pad=10,X=function(v){return pad+v/mx*(w-2*pad)};
      g.clearRect(0,0,w,h);g.font="11px sans-serif";g.fillStyle=c.c1;g.fillRect(X(0),h*.3,X(wt)-X(0)-1,h*.3);g.fillStyle=c.c2;g.fillRect(X(wt),h*.3,X(wt+kv)-X(wt),h*.3);
      lab(g,"weights "+wt+" GB",pad,h*.3-8,c.c1);lab(g,"KV cache "+kv.toFixed(1)+" GB",Math.min(X(wt)+4,w-130),h*.3+h*.3+16,c.c2);
      [[80,"1×80 GB"],[160,"2×80 GB"],[320,"4×80 GB"]].forEach(function(l){if(l[0]>mx)return;g.strokeStyle=c.muted;g.setLineDash([4,4]);g.beginPath();g.moveTo(X(l[0]),h*.2);g.lineTo(X(l[0]),h*.72);g.stroke();g.setLineDash([]);lab(g,l[1],X(l[0]),h*.2-4,c.muted,"center")});
      $("#kvTV").textContent=T;$("#kvBV").textContent=B;$("#kvA").textContent=kv.toFixed(1)+" GB";$("#kvW").textContent=wt+" GB";$("#kvP").textContent=(per/1024).toFixed(0)+" KB"});
    ["kvM","kvD"].forEach(function(id){$("#"+id).addEventListener("change",function(){V.draw()})});["kvT","kvB"].forEach(function(id){$("#"+id).addEventListener("input",function(){V.draw()})})})();

  /* grid vs random */
  (function(){
    var cc=$("#prGrid");if(!cc)return;var seed=5;
    function f(x,y){return Math.exp(-(x-.37)*(x-.37)/.02)+.03*y}
    function runs(n,rr){var pts=[],b=-1;for(var i=0;i<n;i++){var x=rr(),y=rr();pts.push([x,y]);b=Math.max(b,f(x,y))}return [pts,b]}
    function wins(n){var r=rng(77),k=Math.round(Math.sqrt(n)),gb=-1,i,j,w=0;for(i=0;i<k;i++)for(j=0;j<k;j++)gb=Math.max(gb,f((i+.5)/k,(j+.5)/k));
      for(var t=0;t<200;t++){var b=runs(n,r)[1];if(b>gb)w++}return w/2}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,k=Math.floor(Math.sqrt(+$("#gsN").value)),n=k*k,gp=[],gb=-1,i,j;
      for(i=0;i<k;i++)for(j=0;j<k;j++){var x=(i+.5)/k,y=(j+.5)/k;gp.push([x,y]);gb=Math.max(gb,f(x,y))}var R=runs(n,rng(seed)),pw=(w-30)/2;
      g.clearRect(0,0,w,h);g.font="11px sans-serif";
      [["Grid ("+n+" trials)",gp,0],["Random ("+n+" trials)",R[0],1]].forEach(function(P){var x0=10+P[2]*(pw+10),ph=h-40;g.strokeStyle=c.line;g.strokeRect(x0,22,pw,ph);g.fillStyle=rgba(c.c1,.1);g.fillRect(x0+pw*.27,22,pw*.2,ph);
        lab(g,P[0],x0,14,c.muted);P[1].forEach(function(p){g.fillStyle=c.c2;g.beginPath();g.arc(x0+p[0]*pw,22+(1-p[1])*ph,4.5,0,6.3);g.fill();g.fillStyle=c.c1;g.fillRect(x0+p[0]*pw-1,22+ph+4,2,8)});
        lab(g,"important setting →",x0+pw,h-2,c.muted,"right")});
      $("#gsNV").textContent=n;$("#gsA").textContent=gb.toFixed(3);$("#gsB").textContent=R[1].toFixed(3);$("#gsC").textContent=wins(n).toFixed(0)+"%"});
    $("#gsN").addEventListener("input",function(){V.draw()});$("#gsGo").addEventListener("click",function(){seed=Math.floor(Math.random()*9999);V.draw()})})();

  /* drift */
  (function(){
    var cc=$("#prDrift");if(!cc)return;var n=3000,r=rng(21),Z=[],Z2=[],i,an=false,t=0;for(i=0;i<n;i++){Z.push(gauss(r));Z2.push(gauss(r))}var REF=Z.slice().sort(function(a,b){return a-b});
    function stats(sh,sp){var live=Z2.map(function(z){return sh+sp*z}).sort(function(a,b){return a-b}),edges=[-Infinity],k;for(k=1;k<10;k++)edges.push(REF[Math.floor(k*n/10)]);edges.push(Infinity);
      function frac(arr){var f=new Array(10).fill(0);arr.forEach(function(v){for(var b=0;b<10;b++)if(v>=edges[b]&&v<edges[b+1]){f[b]++;break}});return f.map(function(x){return x/n+1e-6})}
      var e=frac(REF),a=frac(live),psi=0;for(k=0;k<10;k++)psi+=(a[k]-e[k])*Math.log(a[k]/e[k]);
      var ia=0,ib=0,ks=0;while(ia<n&&ib<n){if(REF[ia]<=live[ib])ia++;else ib++;ks=Math.max(ks,Math.abs(ia/n-ib/n))}return {psi:psi,ks:ks}}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,sh=+$("#drM").value,sp=+$("#drS").value,nb=30,pad=10,X=function(v){return pad+(v+5)/10*(w-2*pad)},H1=new Array(nb).fill(0),H2=new Array(nb).fill(0),b;
      for(i=0;i<n;i++){b=Math.floor((Z[i]+5)/10*nb);if(b>=0&&b<nb)H1[b]++;b=Math.floor((sh+sp*Z2[i]+5)/10*nb);if(b>=0&&b<nb)H2[b]++}var mx=Math.max.apply(null,H1.concat(H2))*1.1,bw=(w-2*pad)/nb;
      g.clearRect(0,0,w,h);[[H1,rgba(c.muted,.55)],[H2,rgba(c.c1,.6)]].forEach(function(P){g.fillStyle=P[1];P[0].forEach(function(v,k){var bh=v/mx*(h-24);g.fillRect(pad+k*bw+1,h-16-bh,bw-2,bh)})});
      g.strokeStyle=c.line;g.beginPath();g.moveTo(pad,h-16);g.lineTo(w-pad,h-16);g.stroke();g.font="11px sans-serif";lab(g,"reference",pad,12,c.muted);lab(g,"live",pad+70,12,c.c1);
      var s=stats(sh,sp);$("#drMV").textContent=sh.toFixed(1);$("#drSV").textContent=sp.toFixed(1);$("#drP").textContent=s.psi.toFixed(3);$("#drK").textContent=s.ks.toFixed(3);$("#drT").textContent=s.psi>.25?"Act: significant drift":s.psi>.1?"Watch: moderate drift":"Stable"});
    ["drM","drS"].forEach(function(id){$("#"+id).addEventListener("input",function(){V.draw()})});
    $("#drAn").addEventListener("click",function(){an=!an;this.textContent=an?"Stop":"Animate drift"});
    var L=loopOn(cc,function(dt){if(an){t+=dt;$("#drM").value=(1.8*Math.sin(t/1800)).toFixed(2);V.draw()}});if(!rm)L.set(true)})();

  /* A/B */
  (function(){
    var cc=$("#prAb");if(!cc)return;
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,p1=+$("#abP").value/100,p2=p1*(1+ +$("#abL").value/100),n=Math.pow(10,+$("#abN").value),d=p2-p1,pb=(p1+p2)/2,s0=Math.sqrt(2*pb*(1-pb)/n),s1=Math.sqrt((p1*(1-p1)+p2*(1-p2))/n),crit=1.96*s0,
        pow=1-Phi((crit-d)/s1),req=Math.pow(1.96*Math.sqrt(2*pb*(1-pb))+.8416*Math.sqrt(p1*(1-p1)+p2*(1-p2)),2)/(d*d),lo=-4*s0-.2*d,hi=d+4*s1+.2*d,X=function(v){return 10+(v-lo)/(hi-lo)*(w-20)},i;
      function pdf(v,m,s){return Math.exp(-.5*((v-m)/s)*((v-m)/s))/s}var mx=Math.max(pdf(0,0,s0),pdf(d,d,s1))*1.15,Y=function(v){return h-20-v/mx*(h-34)};
      g.clearRect(0,0,w,h);g.font="11px sans-serif";
      [[0,s0,c.muted,false],[d,s1,c.c1,true]].forEach(function(P){g.beginPath();for(i=0;i<=240;i++){var v=lo+(hi-lo)*i/240,y=Y(pdf(v,P[0],P[1]));i?g.lineTo(X(v),y):g.moveTo(X(v),y)}g.strokeStyle=P[2];g.lineWidth=2.4;g.stroke();
        if(P[3]){g.lineTo(X(hi),Y(0));g.lineTo(X(lo),Y(0));g.save();g.beginPath();g.rect(X(crit),0,w,h);g.clip();g.beginPath();for(i=0;i<=240;i++){var v2=lo+(hi-lo)*i/240;i?g.lineTo(X(v2),Y(pdf(v2,P[0],P[1]))):g.moveTo(X(v2),Y(pdf(v2,P[0],P[1])))}g.lineTo(X(hi),Y(0));g.lineTo(X(lo),Y(0));g.fillStyle=rgba(c.c1,.25);g.fill();g.restore()}});
      g.strokeStyle=c.c2;g.setLineDash([5,4]);g.beginPath();g.moveTo(X(crit),8);g.lineTo(X(crit),h-20);g.stroke();g.setLineDash([]);lab(g,"significance threshold",X(crit)+5,16,c.c2);lab(g,"observed difference in rate →",w-10,h-4,c.muted,"right");
      $("#abPV").textContent=(p1*100).toFixed(1);$("#abLV").textContent=$("#abL").value;$("#abNV").textContent=Math.round(n).toLocaleString();$("#abW").textContent=(pow*100).toFixed(0)+"%";$("#abR").textContent=Math.ceil(req).toLocaleString();$("#abZ").textContent=(d/s0).toFixed(2)});
    ["abP","abL","abN"].forEach(function(id){$("#"+id).addEventListener("input",function(){V.draw()})})})();
})();
