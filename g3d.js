/* Tiny 3D engine on a 2D canvas: orbit camera, perspective, depth-sorted dots, lines, quads, boxes and labels. */
(function(){
  window.G3=function(canvas,render,opt){
    opt=opt||{};
    var S={yaw:opt.yaw!=null?opt.yaw:0.7,pitch:opt.pitch!=null?opt.pitch:0.35,dist:opt.dist||4,fov:opt.fov||1.5,spin:opt.spin!=null?opt.spin:0.0002,zoom:opt.zoom||1,drag:false,user:false,q:[],g:null,w:0,h:0,c:null,t:0};
    canvas.classList.add("g3");
    var lastX=0,lastY=0;
    canvas.addEventListener("pointerdown",function(e){S.drag=true;S.user=true;lastX=e.clientX;lastY=e.clientY;try{canvas.setPointerCapture(e.pointerId)}catch(x){}});
    canvas.addEventListener("pointermove",function(e){if(!S.drag)return;S.yaw+=(e.clientX-lastX)*0.008;S.pitch=clamp(S.pitch+(e.clientY-lastY)*0.008,-1.35,1.35);lastX=e.clientX;lastY=e.clientY;S.dirty=true});
    function up(){S.drag=false}
    canvas.addEventListener("pointerup",up);canvas.addEventListener("pointercancel",up);
    S.P=function(x,y,z){
      var cy=Math.cos(S.yaw),sy=Math.sin(S.yaw),cp=Math.cos(S.pitch),sp=Math.sin(S.pitch);
      var x1=x*cy+z*sy,z1=-x*sy+z*cy,y1=y*cp-z1*sp,z2=y*sp+z1*cp;
      var d=S.dist+z2,k=S.fov*Math.min(S.w,S.h)*0.5/Math.max(d,0.2)*S.zoom;
      return {x:S.w/2+x1*k*(S.dist/2.7),y:S.h/2-y1*k*(S.dist/2.7),z:z2,k:k*(S.dist/2.7)}};
    function fog(z){return clamp(1.15-(z+1.6)/(S.dist*1.6),0.25,1)}
    S.dot=function(p,r,col,a){var q=S.P(p[0],p[1],p[2]);S.q.push({z:q.z,f:function(g){g.globalAlpha=(a==null?1:a)*fog(q.z);g.fillStyle=col;g.beginPath();g.arc(q.x,q.y,Math.max(.6,r*q.k*.02),0,6.2832);g.fill()}})};
    S.line=function(a,b,col,wd,al){var p=S.P(a[0],a[1],a[2]),q=S.P(b[0],b[1],b[2]);S.q.push({z:(p.z+q.z)/2,f:function(g){g.globalAlpha=(al==null?1:al)*fog((p.z+q.z)/2);g.strokeStyle=col;g.lineWidth=wd||1;g.beginPath();g.moveTo(p.x,p.y);g.lineTo(q.x,q.y);g.stroke()}})};
    S.quad=function(pts,fill,stroke,al,glow){var ps=pts.map(function(p){return S.P(p[0],p[1],p[2])}),z=0;ps.forEach(function(p){z+=p.z});z/=ps.length;
      S.q.push({z:z,f:function(g){g.globalAlpha=(al==null?1:al)*fog(z);g.beginPath();ps.forEach(function(p,i){i?g.lineTo(p.x,p.y):g.moveTo(p.x,p.y)});g.closePath();if(fill){g.fillStyle=fill;g.fill()}if(stroke){g.strokeStyle=stroke;g.lineWidth=1.2;g.stroke()}}})};
    S.box=function(cx,cy,cz,sx,sy,sz,fill,stroke,al){var x0=cx-sx/2,x1=cx+sx/2,y0=cy-sy/2,y1=cy+sy/2,z0=cz-sz/2,z1=cz+sz/2;
      [[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],[[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],[[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],[[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]],[[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],[[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]]].forEach(function(f){S.quad(f,fill,stroke,al)})};
    S.text=function(p,str,col,size,align){var q=S.P(p[0],p[1],p[2]);S.q.push({z:q.z-0.3,f:function(g){g.globalAlpha=Math.max(.5,fog(q.z));g.fillStyle=col;g.font="500 "+(size||11)+"px "+getComputedStyle(document.body).fontFamily;g.textAlign=align||"center";g.fillText(str,q.x,q.y);g.textAlign="left"}})};
    S.ring=function(c,r,n,col,wd,al,axis){var prev=null;for(var i=0;i<=n;i++){var a=i/n*6.2832,p=axis==="x"?[c[0],c[1]+Math.cos(a)*r,c[2]+Math.sin(a)*r]:axis==="z"?[c[0]+Math.cos(a)*r,c[1]+Math.sin(a)*r,c[2]]:[c[0]+Math.cos(a)*r,c[1],c[2]+Math.sin(a)*r];if(prev)S.line(prev,p,col,wd,al);prev=p}};
    var V=cv(canvas,function(o,c){
      S.g=o.g;S.w=o.w;S.h=o.h;S.c=c;S.q=[];o.g.clearRect(0,0,o.w,o.h);
      render(S,c,o.g,o.w,o.h);
      S.q.sort(function(a,b){return b.z-a.z});
      S.q.forEach(function(it){it.f(o.g)});o.g.globalAlpha=1});
    S.draw=function(){V.draw()};
    S.loop=loopOn(canvas,function(dt){S.t+=dt;if(!S.user&&!rm)S.yaw+=dt*S.spin;if(opt.tick)opt.tick(dt,S);V.draw()});
    if(rm){S.draw()}else S.loop.set(true);
    S.v=V;return S};
})();
