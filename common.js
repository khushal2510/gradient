/* Gradient shared helpers: theme, header/footer, code blocks, TOC, progress, canvas + loop utilities */
(function(){
  var $=function(s,r){return (r||document).querySelector(s)},$$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
  window.$=$;window.$$=$$;
  window.store=function(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}};
  window.clamp=function(x,a,b){return Math.max(a,Math.min(b,x))};
  window.esc=function(s){return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})};
  window.rng=function(seed){var a=seed>>>0;return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}};
  window.gauss=function(r){var u=0,v=0;while(!u)u=r();v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)};
  window.rm=store("g-calm")==="1";
  window.autoRun=function(id){var b=$("#"+id);if(!b)return;var d=b.closest(".demo")||b;if(!("IntersectionObserver" in window)||rm)return;var o=new IntersectionObserver(function(en){if(en[0].isIntersecting){o.disconnect();setTimeout(function(){b.click()},350)}},{threshold:.35});o.observe(d)};

  var PAGES=[["index.html","Home","index"],["ml.html","ML","ml"],["nn.html","Neural nets","nn"],["deep.html","Vision","deep"],["llm.html","LLMs","llm"],["ft.html","Fine-tuning","ft"],["app.html","Applied LLMs","app"],["gen.html","Generative","gen"],["rl.html","RL","rl"],["prod.html","Production","prod"],["play.html","Playbook","play"]];
  window.GPAGES=PAGES;
  var PAGE=window.PAGE||"index";

  /* theme */
  var root=document.documentElement;function isDark(){return true}

  /* colours for canvases */
  window.C=function(){var s=getComputedStyle(root),g=function(n){return s.getPropertyValue(n).trim()};return {ink:g("--ink"),muted:g("--muted"),line:g("--line"),bg:g("--bg"),surface:g("--surface"),surface2:g("--surface2"),accent:g("--accent"),c1:g("--c1"),c2:g("--c2"),c3:g("--c3"),c4:g("--c4"),warn:g("--warn"),dark:isDark()}};
  window.rgba=function(hex,a){hex=hex.replace("#","");if(hex.length===3)hex=hex.replace(/./g,"$&$&");var n=parseInt(hex,16);return "rgba("+(n>>16)+","+(n>>8&255)+","+(n&255)+","+a+")"};
  window.rgbOf=function(hex){hex=hex.replace("#","");if(hex.length===3)hex=hex.replace(/./g,"$&$&");var n=parseInt(hex,16);return [n>>16,n>>8&255,n&255]};

  /* canvas wrapper: handles hi-dpi sizing, theme change and resize */
  var views=[];
  window.cv=function(canvas,draw){
    var o={canvas:canvas,g:null,w:0,h:0};
    function size(){var r=canvas.getBoundingClientRect();if(!r.width||!r.height)return false;var d=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);o.g=canvas.getContext("2d");o.g.setTransform(d,0,0,d,0,0);o.w=r.width;o.h=r.height;return true}
    o.redraw=function(){if(size())draw(o,C())};
    o.draw=function(){if(!o.g&&!size())return;draw(o,C())};
    views.push(o.redraw);o.redraw();return o};
  var rt;function redrawAll(){clearTimeout(rt);rt=setTimeout(function(){views.forEach(function(f){f()})},80)}
  window.addEventListener("resize",redrawAll);
  window.mouse=function(canvas,e){var r=canvas.getBoundingClientRect(),t=e.touches?e.touches[0]:e;return {x:t.clientX-r.left,y:t.clientY-r.top}};

  /* animation loop that only runs while its element is on screen */
  window.loopOn=function(el,fn){
    var L={active:false,vis:true,raf:0,last:0};
    function tick(t){L.raf=0;if(!(L.active&&L.vis))return;var dt=Math.min(64,t-(L.last||t));L.last=t;fn(dt,t);L.raf=requestAnimationFrame(tick)}
    L.kick=function(){if(!L.raf&&L.active&&L.vis){L.last=0;L.raf=requestAnimationFrame(tick)}};
    L.set=function(a){L.active=a;if(a)L.kick()};
    if("IntersectionObserver" in window){new IntersectionObserver(function(es){L.vis=es[0].isIntersecting;if(L.vis)L.kick()},{rootMargin:"80px"}).observe(el)}
    return L};

  /* header */
  var top=$("#topbar");
  if(top){
    top.innerHTML='<div class="bar"><a class="brand" href="index.html" aria-label="Gradient home"><svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="var(--accent)"/><path d="M5 9c6 0 7 14 13 14 3 0 5-2 9-2" fill="none" stroke="var(--accent-ink)" stroke-width="2.6" stroke-linecap="round"/><circle cx="27" cy="21" r="2.6" fill="var(--accent-ink)"/></svg>Gradient</a>'+
      '<nav class="main" id="mainNav" aria-label="Pages">'+PAGES.map(function(p){return '<a href="'+p[0]+'"'+(p[2]===PAGE?' aria-current="page"':'')+'>'+p[1]+'</a>'}).join("")+'</nav>'+
      '<button class="ibtn" id="motionBtn" type="button" aria-pressed="'+(rm?'true':'false')+'" title="Pause or resume animations" style="font:600 .72rem var(--font-mono);width:auto;padding:0 10px">'+(rm?'Motion off':'Motion on')+'</button><button class="ibtn" id="menuBtn" type="button" aria-label="Menu" aria-expanded="false"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button></div><div class="progress" aria-hidden="true"><i id="progBar"></i></div>';
    var nav=$("#mainNav"),mb=$("#menuBtn");$("#motionBtn").addEventListener("click",function(){store("g-calm",rm?"0":"1");location.reload()});
    mb.addEventListener("click",function(){var o=nav.classList.toggle("open");mb.setAttribute("aria-expanded",o?"true":"false")});
    nav.addEventListener("click",function(e){if(e.target.closest("a"))nav.classList.remove("open")});
  }

  /* footer and prev/next */
  var pn=$("#pn");
  if(pn){var i=PAGES.findIndex(function(p){return p[2]===PAGE}),h="";
    if(i>0)h+='<a class="prev" href="'+PAGES[i-1][0]+'"><small>Previous</small><b>'+esc(PAGES[i-1][1])+'</b></a>';
    if(i<PAGES.length-1&&i>=0)h+='<a class="next" href="'+PAGES[i+1][0]+'"><small>Next</small><b>'+esc(PAGES[i+1][1])+'</b></a>';
    pn.innerHTML=h}
  var ft=$("#foot");
  if(ft)ft.innerHTML='<div class="wrap"><div>Gradient, an interactive guide to machine learning. Demos run in your browser and use small toy data, so they show how ideas behave, not production-scale results.</div><nav aria-label="Footer">'+PAGES.map(function(p){return '<a href="'+p[0]+'">'+p[1]+'</a>'}).join("")+'</nav></div>';

  /* code blocks: highlight and copy */
  var KW="def|class|import|from|as|return|for|in|if|else|elif|while|with|lambda|not|and|or|yield|pass|break|continue|try|except|raise|is|assert|const|let|var|function";
  var LIT="None|True|False|self|np|torch|nn|F|plt|pd";
  function hl(src){
    var s=esc(src).replace(/&quot;/g,'"');
    return s.replace(new RegExp("(#.*$)|(\"\"\"[\\s\\S]*?\"\"\"|'[^'\\n]*'|\"[^\"\\n]*\")|\\b("+KW+")\\b|\\b("+LIT+")\\b|\\b(\\d+\\.?\\d*(?:e-?\\d+)?)\\b|\\b([A-Za-z_]\\w*)(?=\\()","gm"),function(m,c,st,k,l,n,f){
      if(c)return '<span class="c">'+c+'</span>';if(st)return '<span class="s">'+st+'</span>';if(k)return '<span class="k">'+k+'</span>';if(l)return '<span class="n">'+l+'</span>';if(n)return '<span class="n">'+n+'</span>';if(f)return '<span class="f">'+f+'</span>';return m})}
  $$(".code").forEach(function(b){
    var pre=$("pre",b),raw=pre.textContent.replace(/^\n/,"").replace(/\s+$/,"");pre.innerHTML="<code>"+hl(raw)+"</code>";
    var btn=$("button",b);if(btn)btn.addEventListener("click",function(){
      var done=function(){btn.textContent="Copied";setTimeout(function(){btn.textContent="Copy"},1400)};
      var fb=function(){var r=document.createRange();r.selectNodeContents(pre);var s=getSelection();s.removeAllRanges();s.addRange(r);btn.textContent="Selected";setTimeout(function(){btn.textContent="Copy"},1400)};
      try{navigator.clipboard.writeText(raw).then(done,fb)}catch(e){fb()}})});

  /* sections: TOC, learned ticks, reveal */
  var secs=$$(".sec[id]");
  var learn={};try{learn=JSON.parse(store("g-learn")||"{}")||{}}catch(e){learn={}}
  function saveLearn(){store("g-learn",JSON.stringify(learn))}
  function updateProgress(){
    var all=window.GSECS||{},tot=0,done=0;Object.keys(all).forEach(function(p){all[p].forEach(function(s){tot++;if(learn[p+":"+s[0]])done++})});
    var bar=$("#progBar");if(bar&&tot)bar.style.width=(100*done/tot)+"%";
    window.dispatchEvent(new CustomEvent("learnchange"));
    $$(".toc a").forEach(function(a){a.classList.toggle("done",!!learn[PAGE+":"+a.getAttribute("href").slice(1)])})}
  window.GLearn=function(){return learn};
  var toc=$("#toc");
  if(toc&&secs.length){
    toc.innerHTML="<b>On this page</b>"+secs.map(function(s){return '<a href="#'+s.id+'">'+esc($("h2",s).textContent)+'</a>'}).join("");
    if("IntersectionObserver" in window){var links=$$("a",toc);
      var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){links.forEach(function(l){l.classList.toggle("on",l.getAttribute("href")==="#"+e.target.id)})}})},{rootMargin:"-20% 0px -70% 0px"});
      secs.forEach(function(s){io.observe(s)});
      var rv=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add("seen");rv.unobserve(e.target)}})},{rootMargin:"0px 0px -8% 0px"});
      secs.forEach(function(s){rv.observe(s)})}
  }
  $$("input[data-learn]").forEach(function(c){var k=PAGE+":"+c.dataset.learn;c.checked=!!learn[k];c.addEventListener("change",function(){if(c.checked)learn[k]=1;else delete learn[k];saveLearn();updateProgress()})});
  updateProgress();
})();

/* shared drawing helpers */
window.heatmap=function(g,w,h,pred,col1,col2,res,alpha){
  var a=rgbOf(col1),b=rgbOf(col2);res=res||8;
  for(var y=0;y<h;y+=res)for(var x=0;x<w;x+=res){
    var p=pred((x+res/2)/w,1-(y+res/2)/h);p=Math.max(0,Math.min(1,p));
    g.fillStyle="rgba("+Math.round(a[0]+(b[0]-a[0])*p)+","+Math.round(a[1]+(b[1]-a[1])*p)+","+Math.round(a[2]+(b[2]-a[2])*p)+","+(alpha||0.3)+")";
    g.fillRect(x,y,Math.min(res,w-x),Math.min(res,h-y))}};
window.dot=function(g,x,y,r,fill,stroke){g.beginPath();g.arc(x,y,r,0,6.2832);if(fill){g.fillStyle=fill;g.fill()}if(stroke){g.strokeStyle=stroke;g.lineWidth=1.6;g.stroke()}};
window.axes=function(g,w,h,c,pad){pad=pad||0;g.strokeStyle=c.line;g.lineWidth=1;g.beginPath();g.moveTo(pad+.5,0);g.lineTo(pad+.5,h-pad);g.lineTo(w,h-pad);g.stroke()};
window.blobs=function(kind,n,noise,seed){
  var r=rng(seed),P=[],i,a,t;
  for(i=0;i<n;i++){
    var cls=i%2;
    if(kind==="moons"){t=r()*Math.PI;var x=cls?1-Math.cos(t):Math.cos(t),y=cls?.4-Math.sin(t):Math.sin(t);P.push({x:.3+x*.2+gauss(r)*noise,y:.38+y*.3+gauss(r)*noise,c:cls})}
    else if(kind==="rings"){a=r()*6.2832;var rad=(cls?.17:.36)+gauss(r)*noise*.7;P.push({x:.5+Math.cos(a)*rad,y:.5+Math.sin(a)*rad,c:cls})}
    else if(kind==="xor"){var qx=r()<.5?0:1,qy=cls?1-qx:qx;P.push({x:.27+qx*.46+gauss(r)*noise,y:.27+qy*.46+gauss(r)*noise,c:cls})}
    else{P.push({x:(cls?.65:.35)+gauss(r)*(noise+.04),y:(cls?.62:.4)+gauss(r)*(noise+.04),c:cls})}
  }
  return P};
