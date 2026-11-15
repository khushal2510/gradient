/* Generative model demos */
(function(){
  /* PCA / autoencoder */
  (function(){
    var cc=$("#aeCv");if(!cc)return;var P,seed=8,cx,cy,Sxx,Syy,Sxy,tot,pc;
    function gen(){var r=rng(seed),a=0.3+r()*1.2,i;P=[];var ca=Math.cos(a),sa=Math.sin(a);for(i=0;i<80;i++){var u=gauss(r)*0.22,v=gauss(r)*0.05;P.push([0.5+u*ca-v*sa,0.5+u*sa+v*ca])}
      cx=0;cy=0;P.forEach(function(p){cx+=p[0];cy+=p[1]});cx/=P.length;cy/=P.length;Sxx=Syy=Sxy=0;P.forEach(function(p){var dx=p[0]-cx,dy=p[1]-cy;Sxx+=dx*dx;Syy+=dy*dy;Sxy+=dx*dy});tot=Sxx+Syy;
      pc=0.5*Math.atan2(2*Sxy,Sxx-Syy);if(pc<0)pc+=Math.PI}
    function err(th){var c=Math.cos(th),s=Math.sin(th),v=Sxx*c*c+2*Sxy*c*s+Syy*s*s;return {e:(tot-v)/P.length,v:v/tot}}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,th=+$("#aeA").value*Math.PI/180,co=Math.cos(th),si=Math.sin(th);g.clearRect(0,0,w,h);if(!P)return;
      var X=function(x){return x*w},Y=function(y){return (1-y)*h};
      g.strokeStyle=c.c1;g.lineWidth=2.4;g.beginPath();g.moveTo(X(cx-co*1),Y(cy-si*1));g.lineTo(X(cx+co*1),Y(cy+si*1));g.stroke();
      P.forEach(function(p){var t=(p[0]-cx)*co+(p[1]-cy)*si,qx=cx+t*co,qy=cy+t*si;g.strokeStyle=rgba(c.c2,.5);g.lineWidth=1;g.beginPath();g.moveTo(X(p[0]),Y(p[1]));g.lineTo(X(qx),Y(qy));g.stroke();dot(g,X(qx),Y(qy),2.5,c.c1);dot(g,X(p[0]),Y(p[1]),4,c.c2)})});
    var E=cv($("#aeErr"),function(o,c){var g=o.g,w=o.w,h=o.h,pad=34;g.clearRect(0,0,w,h);if(!P)return;var mx=err(0).e,mn=1e9,k;for(k=0;k<=180;k++){mx=Math.max(mx,err(k*Math.PI/180).e)}
      var X=function(a){return pad+a/180*(w-pad-8)},Y=function(e){return h-24-e/mx*(h-40)};g.strokeStyle=c.line;g.fillStyle=c.muted;g.font="10px sans-serif";g.beginPath();g.moveTo(pad,8);g.lineTo(pad,h-24);g.lineTo(w-6,h-24);g.stroke();g.fillText("angle (degrees) →",w-100,h-6);g.fillText("error",2,14);
      g.strokeStyle=c.c2;g.lineWidth=2.4;g.beginPath();for(k=0;k<=180;k++){var e=err(k*Math.PI/180).e;k?g.lineTo(X(k),Y(e)):g.moveTo(X(k),Y(e))}g.stroke();
      var a=+$("#aeA").value;dot(g,X(a),Y(err(a*Math.PI/180).e),6,c.c1,c.bg);g.strokeStyle=c.ink;g.setLineDash([3,3]);g.beginPath();g.moveTo(X(pc*180/Math.PI),8);g.lineTo(X(pc*180/Math.PI),h-24);g.stroke();g.setLineDash([]);g.fillStyle=c.ink;g.fillText("PC1",X(pc*180/Math.PI)+4,16)});
    function upd(){var a=+$("#aeA").value;$("#aeAV").textContent=a+"°";var r=err(a*Math.PI/180),b=err(pc);$("#aeE").textContent=r.e.toFixed(4);$("#aeV").textContent=(r.v*100).toFixed(1)+"%";$("#aeB").textContent=b.e.toFixed(4);V.draw();E.draw()}
    $("#aeA").addEventListener("input",upd);$("#aeSnap").addEventListener("click",function(){$("#aeA").value=Math.round(pc*180/Math.PI);upd()});$("#aeNew").addEventListener("click",function(){seed=Math.floor(Math.random()*999);gen();upd()});gen();upd()})();

  /* GAN optimal discriminator */
  (function(){
    var cc=$("#ganCv");if(!cc)return;var MR=2,SR=0.8,anim=false,L;
    function pdf(x,m,s){return Math.exp(-0.5*Math.pow((x-m)/s,2))/(s*2.5066)}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,m=+$("#gnM").value,s=+$("#gnS").value,X=function(x){return (x+4)/10*w},Y=function(p){return h-18-p/0.55*(h-34)};g.clearRect(0,0,w,h);
      g.strokeStyle=c.line;g.lineWidth=1;g.beginPath();g.moveTo(0,h-18);g.lineTo(w,h-18);g.stroke();g.fillStyle=c.muted;g.font="10px sans-serif";for(var x=-4;x<=6;x+=2)g.fillText(x,X(x)-3,h-4);
      [[MR,SR,c.c1],[m,s,c.c2]].forEach(function(q){g.beginPath();g.moveTo(X(-4),h-18);for(var i=0;i<=200;i++){var xx=-4+10*i/200;g.lineTo(X(xx),Y(pdf(xx,q[0],q[1])))}g.lineTo(X(6),h-18);g.closePath();g.fillStyle=rgba(q[2],.28);g.fill();g.strokeStyle=q[2];g.lineWidth=2;g.stroke()});
      g.strokeStyle="#fff";g.lineWidth=2.2;g.beginPath();for(var i2=0;i2<=200;i2++){var x2=-4+10*i2/200,pr=pdf(x2,MR,SR),pg=pdf(x2,m,s),d=pr/(pr+pg+1e-12),yy=h-18-d*(h-34);i2?g.lineTo(X(x2),yy):g.moveTo(X(x2),yy)}g.stroke();
      g.setLineDash([4,4]);g.strokeStyle=c.muted;g.beginPath();g.moveTo(0,h-18-0.5*(h-34));g.lineTo(w,h-18-0.5*(h-34));g.stroke();g.setLineDash([]);g.fillText("0.5",w-24,h-18-0.5*(h-34)-4)});
    function upd(){var m=+$("#gnM").value,s=+$("#gnS").value;$("#gnMV").textContent=m.toFixed(1);$("#gnSV").textContent=s.toFixed(2);
      var js=0,dx=0.02;for(var x=-8;x<=10;x+=dx){var p=pdf(x,MR,SR),q=pdf(x,m,s),mm=(p+q)/2;if(p>1e-12)js+=0.5*p*Math.log(p/mm)*dx;if(q>1e-12)js+=0.5*q*Math.log(q/mm)*dx}
      $("#gnJ").textContent=js.toFixed(3)+" nats";var pr=pdf(MR,MR,SR),pg=pdf(MR,m,s);$("#gnD").textContent=(pr/(pr+pg)).toFixed(3);V.draw()}
    L=loopOn(cc,function(){if(!anim)return;var m=+$("#gnM").value,s=+$("#gnS").value;$("#gnM").value=m+(MR-m)*0.03;$("#gnS").value=s+(SR-s)*0.03;upd();if(Math.abs(MR-m)<0.02&&Math.abs(SR-s)<0.02){anim=false;L.set(false);$("#gnGo").textContent="Animate training"}});
    $("#gnGo").addEventListener("click",function(){anim=!anim;this.textContent=anim?"Pause":"Animate training";L.set(anim)});
    $("#gnRe").addEventListener("click",function(){anim=false;L.set(false);$("#gnGo").textContent="Animate training";$("#gnM").value=-2;$("#gnS").value=0.5;upd()});
    ["gnM","gnS"].forEach(function(id){$("#"+id).addEventListener("input",upd)});upd()})();

  /* diffusion 3D */
  (function(){
    var cc=$("#dfCv");if(!cc)return;
    var T=50,AB=[],i;for(i=0;i<=T;i++){var v=Math.cos((i/T+0.008)/1.008*Math.PI/2);AB.push(v*v)}AB[0]=1;AB[T]=0.0004;
    var X0=[],M=320,E=[],cur=[],t=0,running=false,idx=[],tm=0;
    function shape(k){var r=rng(3),a=[],n=240,j;for(j=0;j<n;j++){var u=j/n*6.2832,p;
      if(k==="knot")p=[Math.sin(u)+2*Math.sin(2*u),Math.cos(u)-2*Math.cos(2*u),-Math.sin(3*u)].map(function(v){return v*0.42});
      else if(k==="torus"){var v2=(j*7.3)%6.2832;p=[(1+0.45*Math.cos(v2))*Math.cos(u),0.45*Math.sin(v2)*1.1,(1+0.45*Math.cos(v2))*Math.sin(u)].map(function(v){return v*0.85})}
      else{var s=j%2?Math.PI:0;p=[Math.cos(u*3+s)*0.7,(u/6.2832-0.5)*2.2,Math.sin(u*3+s)*0.7]}a.push(p)}return a}
    function init(){X0=shape($("#dfS").value);var r=rng(21),k;E=[];idx=[];for(k=0;k<M;k++){E.push([gauss(r),gauss(r),gauss(r)]);idx.push(Math.floor(r()*X0.length))}t=0;set(0)}
    function set(s){t=s;var a=Math.sqrt(AB[s]),b=Math.sqrt(1-AB[s]);cur=E.map(function(e,k){var x=X0[idx[k]];return [a*x[0]+b*e[0],a*x[1]+b*e[1],a*x[2]+b*e[2]]});show()}
    function show(){$("#dfT").value=t;$("#dfTV").textContent=t;$("#dfT2").textContent=t;$("#dfSg").textContent=(Math.sqrt(AB[t])*100).toFixed(0)+"%";$("#dfNs").textContent=(Math.sqrt(1-AB[t])*100).toFixed(0)+"%"}
    function ddim(){/* one reverse step from t to t-1 using the exact score of the data points */
      var ab=AB[t],abp=AB[t-1],sa=Math.sqrt(ab),sn=Math.sqrt(Math.max(1-ab,1e-6)),n=X0.length;
      cur=cur.map(function(x){var w=new Float64Array(n),mx=-1e30,j,d;for(j=0;j<n;j++){var p=X0[j];d=-(Math.pow(x[0]-sa*p[0],2)+Math.pow(x[1]-sa*p[1],2)+Math.pow(x[2]-sa*p[2],2))/(2*sn*sn);w[j]=d;if(d>mx)mx=d}
        var s=0,x0=[0,0,0];for(j=0;j<n;j++){w[j]=Math.exp(w[j]-mx);s+=w[j]}for(j=0;j<n;j++){var q=w[j]/s;x0[0]+=q*X0[j][0];x0[1]+=q*X0[j][1];x0[2]+=q*X0[j][2]}
        var eps=[(x[0]-sa*x0[0])/sn,(x[1]-sa*x0[1])/sn,(x[2]-sa*x0[2])/sn],sap=Math.sqrt(abp),snp=Math.sqrt(1-abp);return [sap*x0[0]+snp*eps[0],sap*x0[1]+snp*eps[1],sap*x0[2]+snp*eps[2]]});t--;show()}
    var S=G3(cc,function(S,c){var a=rgbOf(c.c1),b=rgbOf(c.c2);S.line([-1.5,-1.3,-1.5],[1.5,-1.3,-1.5],c.line,1,.5);
      cur.forEach(function(p,k){var f=k/cur.length;S.dot(p,2.6,"rgb("+Math.round(a[0]+(b[0]-a[0])*f)+","+Math.round(a[1]+(b[1]-a[1])*f)+","+Math.round(a[2]+(b[2]-a[2])*f)+")",.95)})},{yaw:0.6,pitch:0.3,dist:5,spin:0.00012,zoom:0.9,tick:function(dt){if(running){tm+=dt;if(tm>110){tm=0;if(t>0)ddim();else{running=false;$("#dfGo").textContent="Generate from noise"}}}}});
    $("#dfT").addEventListener("input",function(){running=false;$("#dfGo").textContent="Generate from noise";set(+this.value);S.draw()});
    $("#dfGo").addEventListener("click",function(){if(running){running=false;this.textContent="Generate from noise";return}
      if(t<T-1)set(T);else{}running=true;this.textContent="Pause"});
    $("#dfRe").addEventListener("click",function(){running=false;$("#dfGo").textContent="Generate from noise";set(0);S.draw()});
    $("#dfS").addEventListener("change",function(){running=false;init();S.draw()});
    init();S.draw()})();

  /* CFG */
  (function(){
    var cc=$("#cfCv");if(!cc)return;
    function pdf(x,m,s){return Math.exp(-0.5*Math.pow((x-m)/s,2))/(s*2.5066)}
    function curves(w){var xs=[],pu=[],pc=[],pg=[],n=300,i,s=0;for(i=0;i<=n;i++){var x=-3.5+7*i/n;xs.push(x);var u=0.5*pdf(x,-1,0.45)+0.5*pdf(x,1,0.45),c=pdf(x,-1,0.45);pu.push(u);pc.push(c);var lg=w*Math.log(c+1e-300)+(1-w)*Math.log(u+1e-300);pg.push(Math.exp(lg));}
      var dx=7/n,tot=0;pg.forEach(function(v){tot+=v*dx});pg=pg.map(function(v){return v/tot});return {xs:xs,pu:pu,pc:pc,pg:pg,dx:dx}}
    var V=cv(cc,function(o,c){var g=o.g,w=o.w,h=o.h,R=curves(+$("#cfW").value),X=function(x){return (x+3.5)/7*w},mx=0;R.pg.forEach(function(v){mx=Math.max(mx,v)});mx=Math.max(mx,1.2);var Y=function(p){return h-16-Math.min(p/mx,1)*(h-30)};g.clearRect(0,0,w,h);
      g.strokeStyle=c.line;g.beginPath();g.moveTo(0,h-16);g.lineTo(w,h-16);g.stroke();g.fillStyle=c.muted;g.font="10px sans-serif";for(var x=-3;x<=3;x++)g.fillText(x,X(x)-3,h-3);
      [[R.pu,c.muted,1.6],[R.pc,c.c1,2],[R.pg,c.c2,2.8]].forEach(function(q){g.strokeStyle=q[1];g.lineWidth=q[2];g.beginPath();q[0].forEach(function(v,i){i?g.lineTo(X(R.xs[i]),Y(v)):g.moveTo(X(R.xs[i]),Y(v))});g.stroke()})});
    function upd(){var w=+$("#cfW").value;$("#cfWV").textContent=w.toFixed(1);var R=curves(w),m=0,v=0;R.pg.forEach(function(p,i){m+=p*R.xs[i]*R.dx});R.pg.forEach(function(p,i){v+=p*Math.pow(R.xs[i]-m,2)*R.dx});$("#cfS").textContent=Math.sqrt(v).toFixed(3);$("#cfM").textContent=m.toFixed(3);V.draw()}
    $("#cfW").addEventListener("input",upd);upd()})();
})();
autoRun('gnGo');
autoRun('dfGo');
