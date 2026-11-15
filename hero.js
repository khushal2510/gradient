/* Hero artwork: small looping canvases, one per page (data-art). Pauses off screen; static frame for reduced motion. */
(function(){
  var cvEl=document.getElementById("heroCv");if(!cvEl)return;
  var art=cvEl.dataset.art,T=0,R=rng(4),P=[],i;
  for(i=0;i<46;i++){var cl=i%2;P.push({x:(cl?.65:.35)+gauss(R)*.11,y:(cl?.35:.65)+gauss(R)*.11,c:cl})}
  var A={
    net:function(g,w,h,c,t){var L=[3,5,5,2],xs=L.map(function(_,k){return w*(.14+.72*k/(L.length-1))}),nodes=L.map(function(n,k){var a=[];for(var j=0;j<n;j++)a.push([xs[k],h*(.5+(j-(n-1)/2)*.17)]);return a});
      g.lineWidth=1;for(var k=0;k<L.length-1;k++)nodes[k].forEach(function(a,ai){nodes[k+1].forEach(function(b,bi){g.strokeStyle=rgba(c.muted,.28);g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke();
        var ph=((t*.00055+(ai*7+bi*3+k*5)*.13)%1),px=a[0]+(b[0]-a[0])*ph,py=a[1]+(b[1]-a[1])*ph;dot(g,px,py,2.6,rgba(c.accent,.9))})});
      nodes.forEach(function(col,k){col.forEach(function(p,j){dot(g,p[0],p[1],10,c.surface,k===3?c.c2:c.accent)})})},
    scatter:function(g,w,h,c,t){var a=Math.sin(t*.0004)*.5;P.forEach(function(p){dot(g,p.x*w,(1-p.y)*h,5,p.c?c.c2:c.c1)});
      g.strokeStyle=c.ink;g.lineWidth=2;g.beginPath();var cx=w*.5,cy=h*.5,dx=Math.cos(.78+a),dy=-Math.sin(.78+a);g.moveTo(cx-dx*w,cy-dy*w);g.lineTo(cx+dx*w,cy+dy*w);g.stroke()},
    loss:function(g,w,h,c,t){var f=function(x){return .55+.35*Math.sin(x*5)*Math.exp(-x*2.2)+x*.02};g.strokeStyle=c.line;g.lineWidth=1;g.beginPath();g.moveTo(8,h-8);g.lineTo(w-8,h-8);g.stroke();
      g.strokeStyle=c.accent;g.lineWidth=3;g.beginPath();for(var k=0;k<=120;k++){var x=k/120,y=1-(1.1-f(x*1.6)*1);k?g.lineTo(10+x*(w-20),h-14-(1-f(x*1.6)*1)*0+ -((1-f(x*1.6)))*(h-40)*0):g.moveTo(10,h-14)}g.stroke();
      var u=(t*.00018)%1,fx=u*1.6,yy=function(v){return 24+ (1-(1-f(v)))*(h-60)};g.strokeStyle=c.accent;g.beginPath();for(k=0;k<=120;k++){var xx=k/120*1.6;k?g.lineTo(10+xx/1.6*(w-20),yy(xx)):g.moveTo(10,yy(xx))}g.stroke();
      dot(g,10+u*(w-20),yy(fx),9,c.c2,c.bg)},
    conv:function(g,w,h,c,t){var n=9,s=Math.min(w,h)*.8/n,ox=(w-s*n)/2,oy=(h-s*n)/2,k=Math.floor(t*.0025)%49,kx=k%7,ky=Math.floor(k/7);
      for(var y=0;y<n;y++)for(var x=0;x<n;x++){var v=(Math.sin(x*1.3)+Math.cos(y*1.7)+Math.sin((x+y)*.7))/3;g.fillStyle=rgba(c.c1,.12+.5*Math.abs(v));g.fillRect(ox+x*s+1,oy+y*s+1,s-2,s-2)}
      g.strokeStyle=c.c2;g.lineWidth=3;g.strokeRect(ox+kx*s,oy+ky*s,s*3,s*3)},
    attn:function(g,w,h,c,t){var n=7,xs=[];for(var k=0;k<n;k++)xs.push(w*(.1+.8*k/(n-1)));var q=Math.floor(t*.0007)%n;
      for(k=0;k<n;k++){var a=.1+.9*Math.pow(Math.abs(Math.sin(k*1.9+q*2.3)),3);g.strokeStyle=rgba(c.accent,a);g.lineWidth=1+a*4;g.beginPath();g.moveTo(xs[q],h*.68);g.bezierCurveTo(xs[q],h*.15,xs[k],h*.15,xs[k],h*.32);g.stroke()}
      for(k=0;k<n;k++){g.fillStyle=k===q?c.c2:c.surface2;g.strokeStyle=c.line;g.lineWidth=1;g.beginPath();g.roundRect?g.roundRect(xs[k]-w*.05,h*(k===q?.68:.32)-14,w*.1,28,6):g.rect(xs[k]-w*.05,h*.32-14,w*.1,28);g.fill();g.stroke()}},
    diffuse:function(g,w,h,c,t){var u=(Math.sin(t*.0006)+1)/2,r2=rng(9);for(var k=0;k<150;k++){var a=k/150*6.2832,x0=Math.cos(a)*.3,y0=Math.sin(a)*.3,nx=gauss(r2)*.35,ny=gauss(r2)*.35,ab=1-u,x=Math.sqrt(ab)*x0+Math.sqrt(1-ab)*nx,y=Math.sqrt(ab)*y0+Math.sqrt(1-ab)*ny;dot(g,w/2+x*h*1.1,h/2+y*h*1.1,3.4,k%2?c.c1:c.c3)}},
    grid:function(g,w,h,c,t){var n=7,s=Math.min(w,h)*.84/n,ox=(w-s*n)/2,oy=(h-s*n)/2,path=[[0,6],[1,6],[2,6],[2,5],[3,5],[4,5],[4,4],[4,3],[5,3],[6,3],[6,2],[6,1],[6,0]],st=Math.floor(t*.003)%(path.length+4),p=path[Math.min(st,path.length-1)];
      for(var y=0;y<n;y++)for(var x=0;x<n;x++){g.fillStyle=c.surface2;g.fillRect(ox+x*s+1,oy+y*s+1,s-2,s-2)}
      for(var q=0;q<=Math.min(st,path.length-1);q++){g.fillStyle=rgba(c.accent,.25);g.fillRect(ox+path[q][0]*s+1,oy+path[q][1]*s+1,s-2,s-2)}
      g.fillStyle=c.c3;g.fillRect(ox+6*s+1,oy+1,s-2,s-2);dot(g,ox+(p[0]+.5)*s,oy+(p[1]+.5)*s,s*.3,c.c2)},
    checks:function(g,w,h,c,t){var cols=[c.c1,c.c2,c.c3];cols.forEach(function(col,k){g.strokeStyle=col;g.lineWidth=2.6;g.beginPath();for(var j=0;j<=100;j++){var x=j/100,y=.18+.6*Math.exp(-x*(2+k*1.8))+(k===1?.22*x*x:0)+(k===2?.04*Math.sin(j*.9+t*.002)*Math.exp(-x):0);var px=14+x*(w-28),py=h-14-(1-y)*(h-34)*1.05;j?g.lineTo(px,py):g.moveTo(px,py)}g.stroke()})}
  };
  var fn=A[art]||A.net,V=cv(cvEl,function(o,c){var g=o.g;g.clearRect(0,0,o.w,o.h);fn(g,o.w,o.h,c,T)});
  var L=loopOn(cvEl,function(dt){T+=dt;V.draw()});
  if(rm){T=3000;V.draw()}else L.set(true);
})();
