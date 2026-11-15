/* Vision and sequence demos */
(function(){
  /* convolution */
  (function(){
    var ci=$("#cvIn");if(!ci)return;
    var N=10,img=[],hover=null,paint=null;
    var K={vert:[[-1,0,1],[-2,0,2],[-1,0,1]],horiz:[[-1,-2,-1],[0,0,0],[1,2,1]],blur:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],sharp:[[0,-1,0],[-1,5,-1],[0,-1,0]],ident:[[0,0,0],[0,1,0],[0,0,0]]};
    var P={seven:["0000000000","0111111100","0111111100","0000001100","0000011000","0000110000","0001100000","0001100000","0001100000","0000000000"],plus:["0000000000","0000110000","0000110000","0000110000","0111111110","0111111110","0000110000","0000110000","0000110000","0000000000"],box:["0000000000","0111111110","0100000010","0100000010","0100000010","0100000010","0100000010","0100000010","0111111110","0000000000"],blank:new Array(10).fill("0000000000")};
    function load(){img=P[$("#cvP").value].map(function(r){return r.split("").map(Number)})}
    load();
    function conv(){var k=K[$("#cvK").value],relu=$("#cvRelu").getAttribute("aria-pressed")==="true",o=[];for(var i=0;i<8;i++){o.push([]);for(var j=0;j<8;j++){var s=0;for(var a=0;a<3;a++)for(var b=0;b<3;b++)s+=k[a][b]*img[i+a][j+b];o[i].push(relu?Math.max(0,s):s)}}return o}
    function pool(o){var p=[];for(var i=0;i<4;i++){p.push([]);for(var j=0;j<4;j++)p[i].push(Math.max(o[2*i][2*j],o[2*i][2*j+1],o[2*i+1][2*j],o[2*i+1][2*j+1]))}return p}
    function grid(o,c,M,vals,fmt,hl){var g=o.g,w=o.w,s=w/M,mx=0,i,j;vals.forEach(function(r){r.forEach(function(v){mx=Math.max(mx,Math.abs(v))})});mx=mx||1;g.clearRect(0,0,w,o.h);
      for(i=0;i<M;i++)for(j=0;j<M;j++){var v=vals[i][j],t=Math.abs(v)/mx;g.fillStyle=fmt==="bin"?(v?c.c1:c.surface2):(v>=0?rgba(c.c1,.1+.9*t):rgba(c.c2,.1+.9*t));g.fillRect(j*s+1,i*s+1,s-2,s-2)}
      if(hl){g.strokeStyle=c.c2;g.lineWidth=2.5;g.strokeRect(hl[1]*s,hl[0]*s,hl[3]*s,hl[2]*s)}}
    var Vi=cv(ci,function(o,c){grid(o,c,N,img,"bin",hover?[hover[0],hover[1],3,3]:null)});
    var Vo=cv($("#cvOut"),function(o,c){grid(o,c,8,conv(),"val",hover?[hover[0],hover[1],1,1]:null)});
    var Vp=cv($("#cvPool"),function(o,c){grid(o,c,4,pool(conv()),"val",hover?[Math.floor(hover[0]/2),Math.floor(hover[1]/2),1,1]:null)});
    function all(){Vi.draw();Vo.draw();Vp.draw()}
    function ktxt(){var k=K[$("#cvK").value];$("#cvKTxt").textContent=k.map(function(r){return r.map(function(v){return (Math.round(v*100)/100).toString().padStart(6)}).join(" ")}).join("\n")}
    function cell(c,e){var m=mouse(c,e),r=c.getBoundingClientRect();return [Math.floor(m.y/r.height*(c===ci?N:c.id==="cvOut"?8:4)),Math.floor(m.x/r.width*(c===ci?N:c.id==="cvOut"?8:4))]}
    ci.addEventListener("pointerdown",function(e){var q=cell(ci,e);paint=img[q[0]][q[1]]?0:1;img[q[0]][q[1]]=paint;all();ci.setPointerCapture(e.pointerId)});
    ci.addEventListener("pointermove",function(e){if(paint===null)return;var q=cell(ci,e);if(q[0]>=0&&q[0]<N&&q[1]>=0&&q[1]<N&&img[q[0]][q[1]]!==paint){img[q[0]][q[1]]=paint;all()}});
    ci.addEventListener("pointerup",function(){paint=null});
    var out=$("#cvOut");
    function hv(e){var q=cell(out,e);q=[clamp(q[0],0,7),clamp(q[1],0,7)];hover=q;var k=K[$("#cvK").value],s=0,parts=[];for(var a=0;a<3;a++)for(var b=0;b<3;b++){var v=k[a][b]*img[q[0]+a][q[1]+b];s+=v;if(k[a][b]!==0&&img[q[0]+a][q[1]+b])parts.push((Math.round(k[a][b]*100)/100)+"×1")}
      $("#cvCalc").textContent="Output ("+q[0]+","+q[1]+") = "+(parts.length?parts.join(" + "):"0")+" = "+(Math.round(s*100)/100)+($("#cvRelu").getAttribute("aria-pressed")==="true"&&s<0?"  → ReLU gives 0":"");all()}
    out.addEventListener("pointermove",hv);out.addEventListener("pointerdown",hv);out.addEventListener("pointerleave",function(){hover=null;all()});
    $("#cvK").addEventListener("change",function(){ktxt();all()});$("#cvP").addEventListener("change",function(){load();all()});
    $("#cvRelu").addEventListener("click",function(){var on=this.getAttribute("aria-pressed")!=="true";this.setAttribute("aria-pressed",on);this.textContent="ReLU "+(on?"on":"off");all()});
    load();ktxt();all()})();

  /* 3D CNN volume */
  (function(){
    var cc=$("#cnn3d");if(!cc)return;
    var L=[{n:"Input",sub:"28×28×1",sp:28,ch:1},{n:"Conv 5×5",sub:"24×24×8",sp:24,ch:8},{n:"Pool",sub:"12×12×8",sp:12,ch:8},{n:"Conv 5×5",sub:"8×8×16",sp:8,ch:16},{n:"Pool",sub:"4×4×16",sp:4,ch:16}];
    G3(cc,function(S,c){var gap=1.25,x0=-gap*2.3,t=S.t*0.0009;
      L.forEach(function(l,i){var x=x0+i*gap,s=0.2+l.sp/28*1.25,th=0.05+l.ch*0.028;
        S.box(x,0,0,th,s,s,rgba(i%2?c.c2:c.c1,.16),i%2?c.c2:c.c1,.9);
        S.text([x,-s/2-0.22,0],l.n,c.ink,11);S.text([x,-s/2-0.42,0],l.sub,c.muted,10)});
      var fx=x0+4*gap+0.7;for(var i=0;i<16;i++)S.dot([fx,(i-7.5)*0.085,0],1.5,c.c1,.9);S.text([fx,-0.86,0],"Flatten",c.ink,11);S.text([fx,-1.06,0],"256",c.muted,10);
      var ox=fx+0.62;for(i=0;i<10;i++){var on=Math.floor(t*2)%10===i;S.dot([ox,(i-4.5)*0.1,0],on?3.2:1.8,on?c.c2:c.muted,on?1:.7)}S.text([ox,-0.8,0],"Linear",c.ink,11);S.text([ox,-1.0,0],"10 classes",c.muted,10);
      var a=((t*1.3)%1),s0=0.2+1.25,px=x0-0.03,py=(0.5-a)*s0*0.8,pz=(0.5-((a*7)%1))*s0*0.8;
      S.quad([[px-0.02,py-0.1,pz-0.1],[px-0.02,py+0.1,pz-0.1],[px-0.02,py+0.1,pz+0.1],[px-0.02,py-0.1,pz+0.1]],rgba(c.c2,.6),c.c2,1);
      S.line([px,py,pz],[x0+gap,py*0.85,pz*0.85],c.c2,1.2,.7)},{yaw:0.4,pitch:0.28,dist:6.4,spin:0.00008})})();

  /* shapes */
  (function(){
    if(!$("#shW"))return;
    function n(x){return x>=1e9?(x/1e9).toFixed(2)+" G":x>=1e6?(x/1e6).toFixed(2)+" M":x>=1e3?(x/1e3).toFixed(1)+" k":String(Math.round(x))}
    function upd(){var W=+$("#shW").value,K=+$("#shK").value,S=Math.max(1,+$("#shS").value),P=+$("#shP").value,Ci=+$("#shCi").value,Co=+$("#shCo").value,o=Math.floor((W+2*P-K)/S)+1;
      $("#shO").textContent=o>0?o+" × "+o+" × "+Co:"invalid";$("#shPa").textContent=n(K*K*Ci*Co+Co);$("#shM").textContent=o>0?n(o*o*K*K*Ci*Co):"-"}
    ["shW","shK","shS","shP","shCi","shCo"].forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* vanishing gradient */
  (function(){
    var cc=$("#vgCv");if(!cc)return;
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,pad=40,r=+$("#vgR").value,f=+$("#vgF").value,lo=-6,hi=6,X=function(t){return pad+t/50*(w-pad-10)},Y=function(v){var e=clamp(Math.log10(Math.max(v,1e-12)),lo,hi);return h-22-(e-lo)/(hi-lo)*(h-32)};g.clearRect(0,0,w,h);
      g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";for(var e=lo;e<=hi;e+=3){g.beginPath();g.moveTo(pad,Y(Math.pow(10,e)));g.lineTo(w-6,Y(Math.pow(10,e)));g.stroke();g.fillText("1e"+e,2,Y(Math.pow(10,e))+3)}g.fillText("steps back in time →",w-110,h-6);
      [[r,c.c1],[f,c.c2]].forEach(function(s){g.strokeStyle=s[1];g.lineWidth=2.6;g.beginPath();for(var t=0;t<=50;t++){t?g.lineTo(X(t),Y(Math.pow(s[0],t))):g.moveTo(X(t),Y(1))}g.stroke()})});
    function upd(){var r=+$("#vgR").value,f=+$("#vgF").value;$("#vgRV").textContent=r.toFixed(2);$("#vgFV").textContent=f.toFixed(3);$("#vgA").textContent=Math.pow(r,50).toExponential(1);$("#vgB").textContent=Math.pow(f,50).toExponential(1);V.draw()}
    ["vgR","vgF"].forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* SSM 3D */
  (function(){
    var cc=$("#ssm3d");if(!cc)return;var T=120,H=[],Y=[],U=[],shown=T,tm=0;
    function sim(){var rho=+$("#ssR").value,om=+$("#ssW").value,kind=$("#ssI").value,r=rng(4),h=[0,0,0];H=[];Y=[];U=[];
      var co=Math.cos(om),si=Math.sin(om),B=[1,0.4,0.6],Cv=[1,0.6,0.5],lam3=Math.min(0.995,rho+0.02);
      for(var t=0;t<T;t++){var u=kind==="imp"?(t===4?1:0):kind==="pulse"?(t%30===4?1:0):kind==="step"?(t>=4?0.25:0):(r()<0.15?gauss(r):0);U.push(u);
        var a=rho*(co*h[0]-si*h[1])+B[0]*u,b=rho*(si*h[0]+co*h[1])+B[1]*u,d=lam3*h[2]+B[2]*u;h=[a,b,d];H.push(h.slice());Y.push(Cv[0]*a+Cv[1]*b+Cv[2]*d)}
      $("#ssRV").textContent=rho.toFixed(3);$("#ssWV").textContent=om.toFixed(2);$("#ssH").textContent=rho<1?(Math.log(0.5)/Math.log(rho)).toFixed(1)+" steps":"∞";$("#ssK").textContent=rho<1?Math.ceil(Math.log(0.01)/Math.log(rho))+" steps":"∞";shown=0}
    var S=G3(cc,function(S,c){var sc=0.55;S.line([-1.2,0,0],[1.2,0,0],c.line,1,.8);S.line([0,-1.2,0],[0,1.2,0],c.line,1,.8);S.line([0,0,-1.2],[0,0,1.2],c.line,1,.8);S.text([1.3,0,0],"h₁",c.muted,11);S.text([0,1.3,0],"h₃",c.muted,11);S.text([0,0,1.3],"h₂",c.muted,11);
      if(!H.length)return;var n=Math.min(shown,T);for(var t=1;t<n;t++){var a=H[t-1],b=H[t],f=t/n;S.line([a[0]*sc,a[2]*sc,a[1]*sc],[b[0]*sc,b[2]*sc,b[1]*sc],rgba(c.c2,.2+.8*f),1+2*f,1)}
      if(n>0){var p=H[n-1];S.dot([p[0]*sc,p[2]*sc,p[1]*sc],5,c.c1,1)}},{yaw:0.6,pitch:0.35,dist:4.6,spin:0.00018,tick:function(dt){if(shown<T){tm+=dt;if(tm>25){shown++;tm=0;drawY()}}}});
    var VY=cv($("#ssmY"),function(o,c){var g=o.g,w=o.w,h=o.h;g.clearRect(0,0,w,h);if(!Y.length)return;var mx=0.001;Y.forEach(function(v){mx=Math.max(mx,Math.abs(v))});U.forEach(function(v){mx=Math.max(mx,Math.abs(v))});
      var X=function(t){return 8+t/(T-1)*(w-16)},Yp=function(v){return h/2-v/mx*(h/2-14)};g.strokeStyle=c.line;g.lineWidth=1;g.beginPath();g.moveTo(0,h/2);g.lineTo(w,h/2);g.stroke();
      g.strokeStyle=c.c4;g.lineWidth=1.4;g.beginPath();U.forEach(function(v,t){t?g.lineTo(X(t),Yp(v)):g.moveTo(X(t),Yp(v))});g.stroke();
      g.strokeStyle=c.c1;g.lineWidth=2.4;g.beginPath();for(var t=0;t<Math.min(shown,T);t++){t?g.lineTo(X(t),Yp(Y[t])):g.moveTo(X(t),Yp(Y[t]))}g.stroke()});
    function drawY(){VY.draw()}
    ["ssR","ssW","ssI"].forEach(function(id){$("#"+id).addEventListener("input",function(){sim();drawY()})});
    $("#ssRe").addEventListener("click",function(){shown=0;drawY()});
    sim();if(rm){shown=T}drawY()})();
})();
autoRun('ssRe');
