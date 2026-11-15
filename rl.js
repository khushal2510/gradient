/* Reinforcement learning demos */
(function(){
  var W=6,H=5,NS=W*H,START=4*W+0,GOAL=0*W+5,PIT=1*W+5,WALLS={},STEP=-0.04;
  [[1,3],[2,1],[2,2],[4,2]].forEach(function(p){WALLS[p[1]*W+p[0]]=1});
  var DX=[0,1,0,-1],DY=[-1,0,1,0];
  function mv(s,a){var x=s%W,y=Math.floor(s/W),nx=x+DX[a],ny=y+DY[a];if(nx<0||ny<0||nx>=W||ny>=H||WALLS[ny*W+nx])return s;return ny*W+nx}
  function term(s){return s===GOAL||s===PIT}
  function trans(s,a,slip){var o=[[mv(s,a),1-slip]];if(slip>0){o.push([mv(s,(a+1)%4),slip/2]);o.push([mv(s,(a+3)%4),slip/2])}return o}

  /* shared 3D grid renderer */
  function drawGrid(S,c,val,arrows,agent){
    var cell=0.5,base=-0.55,lo=rgbOf(c.c2),hi=rgbOf(c.c1);
    for(var s=0;s<NS;s++){var x=s%W,y=Math.floor(s/W),cx=(x-(W-1)/2)*cell,cz=(y-(H-1)/2)*cell,v=clamp(val[s],-1,1);
      if(WALLS[s]){S.box(cx,base+0.55,cz,cell*0.92,1.1,cell*0.92,"rgba(120,120,130,.35)","rgba(200,200,210,.5)",.9);continue}
      var hgt=0.06+(v+1)*0.42,t=(v+1)/2,col="rgba("+Math.round(lo[0]+(hi[0]-lo[0])*t)+","+Math.round(lo[1]+(hi[1]-lo[1])*t)+","+Math.round(lo[2]+(hi[2]-lo[2])*t)+",";
      var isG=s===GOAL,isP=s===PIT;S.box(cx,base+hgt/2,cz,cell*0.9,hgt,cell*0.9,col+(isG||isP?".85":".45")+")",isG?c.c1:isP?c.c2:col+".95)",1);
      if(isG)S.text([cx,base+hgt+0.12,cz],"+1",c.c1,12);if(isP)S.text([cx,base+hgt+0.12,cz],"−1",c.c2,12);
      if(s===START&&!agent)S.text([cx,base+hgt+0.12,cz],"start",c.muted,10);
      if(arrows&&!term(s)&&arrows[s]>=0){var a=arrows[s],ty=base+hgt+0.02;S.line([cx,ty,cz],[cx+DX[a]*cell*0.32,ty,cz+DY[a]*cell*0.32],c.ink,2,1);S.dot([cx+DX[a]*cell*0.32,ty,cz+DY[a]*cell*0.32],2.4,c.ink,1)}}
    if(agent!=null){var ax=agent%W,ay=Math.floor(agent/W),vv=clamp(val[agent],-1,1);S.dot([(ax-(W-1)/2)*cell,base+0.06+(vv+1)*0.42+0.1,(ay-(H-1)/2)*cell],6,"#fff",1)}}

  /* value iteration */
  (function(){
    var cc=$("#viCv");if(!cc)return;var V,sweeps,delta,run=true,acc=0;
    function reset(){V=new Array(NS).fill(0);V[GOAL]=1;V[PIT]=-1;sweeps=0;delta=null;info()}
    function qv(s,a,g,slip){var q=0;trans(s,a,slip).forEach(function(t){q+=t[1]*(STEP+g*V[t[0]])});return q}
    function sweep(){var g=+$("#viG").value,slip=+$("#viS").value,nv=V.slice(),d=0;for(var s=0;s<NS;s++){if(WALLS[s]||term(s))continue;var b=-1e9;for(var a=0;a<4;a++)b=Math.max(b,qv(s,a,g,slip));nv[s]=b;d=Math.max(d,Math.abs(b-V[s]))}V=nv;sweeps++;delta=d;info()}
    function pol(){var g=+$("#viG").value,slip=+$("#viS").value;return V.map(function(_,s){if(WALLS[s]||term(s))return -1;var b=-1e9,ba=0;for(var a=0;a<4;a++){var q=qv(s,a,g,slip);if(q>b+1e-9){b=q;ba=a}}return sweeps?ba:-1})}
    function info(){$("#viN").textContent=sweeps;$("#viD").textContent=delta==null?"-":delta.toExponential(1);$("#viVs").textContent=V[START].toFixed(3);$("#viGV").textContent=(+$("#viG").value).toFixed(2);$("#viSV").textContent=(+$("#viS").value).toFixed(2)}
    var S=G3(cc,function(S,c){if(!V)return;drawGrid(S,c,V,pol(),null)},{yaw:0.55,pitch:0.55,dist:5,spin:0.00008,zoom:0.95,tick:function(dt){if(!run)return;acc+=dt;if(acc>260){acc=0;if(delta==null||delta>1e-5)sweep()}}});
    $("#viRun").addEventListener("click",function(){run=!run;this.textContent=run?"Pause":"Run"});
    $("#viStep").addEventListener("click",function(){sweep();S.draw()});
    $("#viReset").addEventListener("click",function(){reset();S.draw()});
    ["viG","viS"].forEach(function(id){$("#"+id).addEventListener("input",function(){reset();S.draw()})});
    reset();$("#viRun").textContent="Pause";S.draw()})();

  /* Q-learning */
  (function(){
    var cc=$("#qlCv");if(!cc)return;var Q,s,steps,ep,hist,wins,run=true,acc=0;
    function reset(){Q=[];for(var i=0;i<NS;i++)Q.push([0,0,0,0]);s=START;steps=0;ep=0;hist=[];wins=[];info()}
    function argmax(q){var b=-1e9,c=[];q.forEach(function(v,i){if(v>b+1e-12){b=v;c=[i]}else if(Math.abs(v-b)<=1e-12)c.push(i)});return c[Math.floor(Math.random()*c.length)]}
    function once(){var a=+$("#qlA").value,e=+$("#qlE").value,g=+$("#qlG").value,act=Math.random()<e?Math.floor(Math.random()*4):argmax(Q[s]),s2=mv(s,act),r=s2===GOAL?1:s2===PIT?-1:STEP,done=term(s2);
      Q[s][act]+=a*(r+(done?0:g*Math.max.apply(null,Q[s2]))-Q[s][act]);s=s2;steps++;
      if(done||steps>=80){ep++;hist.push(steps);wins.push(s===GOAL?1:0);if(hist.length>200)hist.shift();if(wins.length>20)wins.shift();s=START;steps=0;info()}}
    function vals(){return Q.map(function(q,i){return term(i)?(i===GOAL?1:-1):Math.max.apply(null,q)})}
    function pol(){return Q.map(function(q,i){return term(i)||WALLS[i]?-1:(Math.max.apply(null,q)===0&&Math.min.apply(null,q)===0?-1:q.indexOf(Math.max.apply(null,q)))})}
    function info(){$("#qlN").textContent=ep;$("#qlL").textContent=hist.length?hist[hist.length-1]:"-";$("#qlS").textContent=wins.length?Math.round(100*wins.reduce(function(a,b){return a+b},0)/wins.length)+"%":"-";$("#qaV").textContent=(+$("#qlA").value).toFixed(2);$("#qeV").textContent=(+$("#qlE").value).toFixed(2);$("#qgV").textContent=(+$("#qlG").value).toFixed(2)}
    var S=G3(cc,function(S,c){if(!Q)return;drawGrid(S,c,vals(),pol(),s)},{yaw:0.55,pitch:0.55,dist:5,spin:0.00008,zoom:0.95,tick:function(dt){if(!run)return;acc+=dt;while(acc>=22){acc-=22;once()}}});
    var CV=cv($("#qlCurve"),function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!hist)return;g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";g.beginPath();g.moveTo(30,6);g.lineTo(30,h-18);g.lineTo(w-4,h-18);g.stroke();g.fillText("steps",2,12);g.fillText("episodes →",w-70,h-4);[0,40,80].forEach(function(v){var y=h-18-v/80*(h-30);g.fillText(v,8,y+3)});
      g.strokeStyle=c.c2;g.lineWidth=2;g.beginPath();hist.forEach(function(v,i){var x=30+i/Math.max(hist.length-1,1)*(w-36),y=h-18-v/80*(h-30);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()});
    var tc=0;var L=loopOn(cc,function(dt){tc+=dt;if(tc>200){tc=0;CV.draw()}});if(!rm)L.set(true);
    $("#qlRun").addEventListener("click",function(){run=!run;this.textContent=run?"Pause":"Run"});
    $("#qlReset").addEventListener("click",function(){reset();S.draw();CV.draw()});
    ["qlA","qlE","qlG"].forEach(function(id){$("#"+id).addEventListener("input",info)});
    reset();S.draw()})();

  /* bandits */
  (function(){
    var cc=$("#bnCv");if(!cc)return;var R=null,seed=1;
    function sim(arms,T,runs,sd){var best=Math.max.apply(null,arms),r=rng(sd),K=arms.length,out=[0,1,2,3].map(function(){return new Float64Array(T/10)});
      for(var run=0;run<runs;run++)for(var st=0;st<4;st++){var n=new Array(K).fill(0),s=new Array(K).fill(0),reg=0;
        for(var t=0;t<T;t++){var a;if(t<K)a=t;else if(st===0)a=argmax(s.map(function(v,i){return v/n[i]}));
          else if(st===1)a=r()<0.1?Math.floor(r()*K):argmax(s.map(function(v,i){return v/n[i]}));
          else if(st===2)a=argmax(s.map(function(v,i){return v/n[i]+Math.sqrt(2*Math.log(t+1)/n[i])}));
          else a=argmax(s.map(function(v,i){return betaS(r,1+v,1+n[i]-v)}));
          var rew=r()<arms[a]?1:0;n[a]++;s[a]+=rew;reg+=best-arms[a];if((t+1)%10===0)out[st][(t+1)/10-1]+=reg/runs}}
      return out}
    function argmax(a){var b=-1e9,bi=0;a.forEach(function(v,i){if(v>b){b=v;bi=i}});return bi}
    function gammaS(r,k){if(k<1){return gammaS(r,k+1)*Math.pow(r(),1/k)}var d=k-1/3,c=1/Math.sqrt(9*d);for(;;){var x=gauss(r),v=1+c*x;if(v<=0)continue;v=v*v*v;var u=r();if(Math.log(u)<0.5*x*x+d-d*v+d*Math.log(v))return d*v}}
    function betaS(r,a,b){var x=gammaS(r,a),y=gammaS(r,b);return x/(x+y)}
    function run(){var r=rng(seed*97+5),arms=[];for(var i=0;i<5;i++)arms.push(0.1+0.8*r());arms.sort(function(a,b){return b-a});R=sim(arms,1000,40,seed);$("#bnArms").textContent=arms.map(function(a){return a.toFixed(2)}).join(", ");R.forEach(function(c,i){$("#bnR"+i).textContent=c[c.length-1].toFixed(1)});V.draw()}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!R)return;var mx=0;R.forEach(function(a){mx=Math.max(mx,a[a.length-1])});mx=Math.max(mx,1);g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";g.beginPath();g.moveTo(40,6);g.lineTo(40,h-18);g.lineTo(w-4,h-18);g.stroke();g.fillText("regret",2,12);g.fillText("pulls →",w-48,h-4);[0,.5,1].forEach(function(f){g.fillText((mx*f).toFixed(0),8,h-18-f*(h-30)+3)});
      [c.muted,c.c1,c.c2,c.c4].forEach(function(col,i){g.strokeStyle=col;g.lineWidth=2.4;g.beginPath();R[i].forEach(function(v,k){var x=40+k/(R[i].length-1)*(w-46),y=h-18-v/mx*(h-30);k?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()})});
    $("#bnGo").addEventListener("click",function(){seed++;run()});run()})();

  /* REINFORCE */
  (function(){
    var cc=$("#pgCv");if(!cc)return;var MU=[0.2,0.5,0.8],th,H,base=true,run=false,bb;
    function reset(){th=[0,0,0];H=[];bb=0}
    function pi(){var m=Math.max.apply(null,th),e=th.map(function(v){return Math.exp(v-m)}),s=e[0]+e[1]+e[2];return e.map(function(v){return v/s})}
    function step(){var p=pi(),u=Math.random(),a=u<p[0]?0:u<p[0]+p[1]?1:2,r=MU[a]+0.3*gauss(Math.random),b=base?bb:0;for(var k=0;k<3;k++)th[k]+=0.15*(r-b)*((k===a?1:0)-p[k]);bb=0.9*bb+0.1*r;H.push(pi())}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!H)return;g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";g.beginPath();g.moveTo(30,6);g.lineTo(30,h-18);g.lineTo(w-4,h-18);g.stroke();g.fillText("probability",2,12);g.fillText("step →",w-44,h-4);[0,.5,1].forEach(function(v){g.fillText(v,8,h-18-v*(h-30)+3)});
      [c.muted,c.c1,c.c2].forEach(function(col,k){g.strokeStyle=col;g.lineWidth=2.4;g.beginPath();H.forEach(function(p,i){var x=30+i/400*(w-36),y=h-18-p[k]*(h-30);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()})});
    var L=loopOn(cc,function(){if(!run)return;for(var i=0;i<6&&H.length<400;i++)step();$("#pgP").textContent=H.length?(H[H.length-1][2]*100).toFixed(1)+"%":"-";V.draw();if(H.length>=400){run=false;L.set(false);$("#pgRun").textContent="Run 400 steps"}});
    $("#pgRun").addEventListener("click",function(){reset();run=true;this.textContent="Running…";L.set(true)});
    $("#pgBase").addEventListener("click",function(){base=!base;this.setAttribute("aria-pressed",base);this.textContent="Baseline "+(base?"on":"off")});
    reset();V.draw()})();

  /* PPO clip */
  (function(){
    var cc=$("#ppCv");if(!cc)return;
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,e=+$("#ppE").value,X=function(r){return 30+r/2*(w-38)},Y=function(v){return h-18-(v+2.2)/3.7*(h-30)};g.clearRect(0,0,w,h);
      g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";g.beginPath();g.moveTo(30,6);g.lineTo(30,h-18);g.moveTo(30,Y(0));g.lineTo(w-4,Y(0));g.stroke();g.fillText("ratio ρ →",w-52,h-4);g.fillText("objective",2,12);
      [0,0.5,1,1.5,2].forEach(function(r){g.fillText(r,X(r)-6,h-6)});
      [[1,c.c1],[-1,c.c2]].forEach(function(q){var A=q[0];g.strokeStyle=q[1];g.lineWidth=2.8;g.beginPath();for(var i=0;i<=100;i++){var r=2*i/100,cl=clamp(r,1-e,1+e),v=Math.min(r*A,cl*A);i?g.lineTo(X(r),Y(v)):g.moveTo(X(r),Y(v))}g.stroke()});
      g.strokeStyle=c.ink;g.setLineDash([3,3]);[1-e,1+e].forEach(function(r){g.beginPath();g.moveTo(X(r),6);g.lineTo(X(r),h-18);g.stroke()});g.setLineDash([]);g.fillStyle=c.ink;g.fillText("1−ε",X(1-e)-14,16);g.fillText("1+ε",X(1+e)+2,16)});
    $("#ppE").addEventListener("input",function(){$("#ppEV").textContent=(+this.value).toFixed(2);V.draw()});V.draw()})();

  /* Dyna-Q */
  (function(){
    var cc=$("#dyCv");if(!cc)return;var R=null;
    function amax(q,r){var b=-1e9,c=[];for(var i=0;i<4;i++){if(q[i]>b+1e-12){b=q[i];c=[i]}else if(Math.abs(q[i]-b)<=1e-12)c.push(i)}return c[Math.floor(r()*c.length)]}
    function dyna(n,seed,E){var r=rng(seed),Q=[],i,model={},keys=[],out=[];for(i=0;i<NS;i++)Q.push([0,0,0,0]);
      for(var e=0;e<E;e++){var s=START,steps=0;while(steps<300){var a=r()<0.1?Math.floor(r()*4):amax(Q[s],r),s2=mv(s,a),rw=s2===GOAL?1:s2===PIT?-1:STEP,d=term(s2);
          Q[s][a]+=0.5*(rw+(d?0:0.95*Math.max.apply(null,Q[s2]))-Q[s][a]);var k=s*4+a;if(!model[k])keys.push(k);model[k]=[rw,s2,d];
          for(var p=0;p<n;p++){var kk=keys[Math.floor(r()*keys.length)],m=model[kk],ss=Math.floor(kk/4),aa=kk%4;Q[ss][aa]+=0.5*(m[0]+(m[2]?0:0.95*Math.max.apply(null,Q[m[1]]))-Q[ss][aa])}
          s=s2;steps++;if(d)break}out.push(steps)}return out}
    function run(){var E=40,res=[0,10,50].map(function(n){var avg=new Array(E).fill(0);for(var sd=0;sd<6;sd++){dyna(n,sd+1,E).forEach(function(v,i){avg[i]+=v/6})}return avg});R=res;
      [["#dyA",0],["#dyB",1],["#dyC",2]].forEach(function(p){var a=res[p[1]],k=a.findIndex(function(v){return v<=15});$(p[0]).textContent=k<0?"not within 40":(k+1)+" episodes"});V.draw()}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!R)return;var mx=120;g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";g.beginPath();g.moveTo(34,6);g.lineTo(34,h-18);g.lineTo(w-4,h-18);g.stroke();g.fillText("steps",2,12);g.fillText("episodes →",w-64,h-4);[0,40,80,120].forEach(function(v){g.fillText(v,8,h-18-v/mx*(h-30)+3)});
      [c.muted,c.c1,c.c2].forEach(function(col,i){g.strokeStyle=col;g.lineWidth=2.6;g.beginPath();R[i].forEach(function(v,k){var x=34+k/(R[i].length-1)*(w-40),y=h-18-Math.min(v,mx)/mx*(h-30);k?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()})});
    $("#dyGo").addEventListener("click",function(){this.textContent="Running…";var b=this;setTimeout(function(){run();b.textContent="Run again"},30)});
    setTimeout(run,60)})();
})();
autoRun('viRun');
autoRun('bnGo');
autoRun('pgRun');
autoRun('dyGo');
