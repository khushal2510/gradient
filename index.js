(function(){
  var info={ml:["Classical ML","Regression, trees, k-means, bias and variance, evaluation."],nn:["Neural nets","Backprop, optimisers, initialisation and regularisation."],deep:["Vision & sequences","Convolutions, 3D CNN volume, LSTMs and state space models."],llm:["Transformers & LLMs","Tokens, embeddings, attention, a 3D transformer, RAG and LoRA."],ft:["Training & fine-tuning","Pretraining, scaling laws, SFT, LoRA, DPO, GRPO and quantisation."],app:["Applied LLMs","RAG, tool calling, structured output, reasoning, evals and CLIP."],gen:["Generative models","Autoencoders, GANs, 3D diffusion and guidance."],rl:["RL & world models","Value iteration, Q-learning, bandits, PPO and Dyna."],prod:["Production & scaling","FSDP, pipelines, vLLM batching, drift, A/B tests and rollout."],play:["Playbook","A training doctor, LR test, glossary and quiz."]};
  var box=$("#hmCards");if(!box)return;
  function draw(){var L=GLearn(),all=window.GSECS||{},td=0,tn=0;
    box.innerHTML=Object.keys(info).map(function(k,i){var s=all[k]||[],d=0;s.forEach(function(x){if(L[k+":"+x[0]])d++});td+=d;tn+=s.length;
      return '<a class="card" href="'+k+'.html"><span class="no">'+String(i+1).padStart(2,"0")+'</span><h3>'+esc(info[k][0])+'</h3><p>'+esc(info[k][1])+'</p><div class="bar-r" style="grid-template-columns:minmax(0,1fr) 48px"><div class="t"><i style="width:'+(s.length?d/s.length*100:0)+'%"></i></div><span>'+d+"/"+s.length+'</span></div></a>'}).join("");
    $("#hmTot").textContent=td+" of "+tn+" concepts marked as learned"}
  draw();window.addEventListener("learnchange",draw)})();
