export default function(component) {
  const {data, parentElement, setTriggerValue, setStateValue} = component;
  const root = parentElement.querySelector('.uw-assistant-root');
  if (!root._ui) {
    root.innerHTML = `
      <button class="uw-launcher" aria-label="Open AI underwriting assistant" aria-expanded="false" aria-controls="uw-chat-panel">
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 11.5a8 8 0 0 1-8 8H5l-3 3v-11a9 9 0 0 1 18 0Z" stroke="currentColor" stroke-width="1.7"/><path d="M7 11h8M7 15h5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
        <span>AI assistant</span>
      </button>
      <section class="uw-panel" id="uw-chat-panel" role="dialog" aria-modal="false" aria-labelledby="uw-chat-heading" hidden>
        <header class="uw-header"><div><h2 id="uw-chat-heading">Underwriting assistant</h2><p class="uw-loan"></p></div><button type="button" class="uw-close" aria-label="Close AI assistant">×</button></header>
        <div class="uw-context"></div>
        <div class="uw-log" role="log" aria-label="Loan conversation" aria-live="polite" aria-relevant="additions" tabindex="0"></div>
        <footer class="uw-footer"><form class="uw-form"><textarea class="uw-input" rows="2" maxlength="2000" aria-label="Ask about the selected loan" placeholder="Ask about this loan’s income…"></textarea><button class="uw-send" type="submit" aria-label="Send question">↑</button></form><div class="uw-notice" role="alert"></div><p class="uw-disclaimer">Synthetic data · Evidence support only. Human review required.</p></footer>
      </section>`;
    root._ui = {open: !!data.is_open, signature: '', context: data.context_id, drafts: {}, busy: false};
  }
  const ui = root._ui;
  const q = selector => root.querySelector(selector);
  const panel = q('.uw-panel'), launcher = q('.uw-launcher'), input = q('.uw-input'), log = q('.uw-log');
  const isMobile = () => window.matchMedia('(max-width: 600px)').matches;
  const applyOpen = (open, moveFocus=true) => {
    ui.open = open;
    panel.hidden = !open;
    panel.setAttribute('aria-modal', String(isMobile()));
    launcher.setAttribute('aria-expanded', String(open));
    launcher.setAttribute('aria-label', open ? 'Minimize AI underwriting assistant' : 'Open AI underwriting assistant');
    if (moveFocus) requestAnimationFrame(() => (open ? input : launcher).focus());
  };
  const toggle = open => { applyOpen(open); setStateValue('is_open', open); };
  launcher.onclick = () => toggle(!ui.open);
  q('.uw-close').onclick = () => toggle(false);
  if (ui.context !== data.context_id) {
    ui.drafts[ui.context] = input.value;
    ui.context = data.context_id;
    input.value = ui.drafts[data.context_id] || '';
    ui.signature = '';
    q('.uw-notice').textContent = '';
  }
  q('.uw-loan').textContent = `${data.loan_id} · ${data.employment_type.replaceAll('_', ' ')}`;
  q('.uw-context').textContent = `${data.source_label} · ${data.view} · Selected-loan access only`;
  const signature = JSON.stringify([data.messages, data.busy]);
  const wasBusy = ui.busy;
  ui.busy = data.busy;
  if (signature !== ui.signature) {
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 90;
    log.replaceChildren();
    if (!data.messages.length) {
      const empty = document.createElement('div'); empty.className = 'uw-empty';
      empty.innerHTML = '<h3>A second look,<br>grounded in evidence.</h3><p>Review the selected loan’s income facts and available guidelines. Start with a question below.</p>';
      data.prompts.forEach(prompt => {
        const button = document.createElement('button'); button.className='uw-suggestion'; button.type='button'; button.textContent=prompt;
        button.onclick=()=>{input.value=prompt; input.focus();}; empty.append(button);
      });
      log.append(empty);
    }
    for (const message of data.messages) {
      const article = document.createElement('article'); article.className=`uw-message ${message.role}`;
      const label=document.createElement('div'); label.className='uw-message-label'; label.textContent=message.role==='user'?'You':'AI assistant';
      const body=document.createElement('div'); body.className='uw-body';
      // Only the Python renderer's escaped, restricted markup is accepted here.
      if (message.role==='assistant') body.innerHTML=message.html; else body.textContent=message.text;
      article.append(label,body);
      if (message.audit) {
        const details=document.createElement('details');
        const summary=document.createElement('summary'); summary.textContent='Evidence audit trail';
        const pre=document.createElement('pre'); pre.textContent=JSON.stringify(message.audit,null,2);
        details.append(summary,pre); article.append(details);
      }
      if (message.state) {
        const status=document.createElement('div'); status.className=`uw-status ${message.state}`;
        status.setAttribute('role',message.state==='error'?'alert':'status');
        status.textContent=message.state==='error'?'Request failed. Check the index and Ollama, then try again.':'Insufficient evidence — review the cited original sources.';
        article.prepend(status);
      }
      log.append(article);
    }
    if (data.busy) {
      const loading=document.createElement('div'); loading.className='uw-status loading'; loading.setAttribute('role','status');
      loading.innerHTML='<span class="uw-pulse" aria-hidden="true"></span>Retrieving evidence and reviewing this loan… You can minimize while I work.'; log.append(loading);
    }
    ui.signature=signature;
    if (nearBottom || !data.messages.length || wasBusy !== data.busy) requestAnimationFrame(()=>{log.scrollTop=log.scrollHeight;});
    if (wasBusy && !data.busy && ui.open && document.activeElement === document.body) input.focus();
  }
  q('.uw-send').disabled=!!data.busy;
  panel.setAttribute('aria-busy',String(!!data.busy));
  const submit = event => {
    event.preventDefault();
    const text=input.value.trim();
    if (!text || ui.busy) return;
    if (text.length>2000) {q('.uw-notice').textContent='Use 2,000 characters or fewer.'; return;}
    q('.uw-notice').textContent='';
    ui.busy=true; q('.uw-send').disabled=true;
    input.value=''; ui.drafts[data.context_id]='';
    const loading=document.createElement('div'); loading.className='uw-status loading'; loading.setAttribute('role','status'); loading.textContent='Sending your question…'; log.append(loading); log.scrollTop=log.scrollHeight;
    setTriggerValue('submit', {context_id:data.context_id, request_id:crypto.randomUUID(), text});
  };
  q('.uw-form').onsubmit=submit;
  input.oninput=()=>{ui.drafts[data.context_id]=input.value;};
  input.onkeydown=event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing)submit(event);};
  const onKey = event => {
    if (!ui.open) return;
    if (event.key==='Escape') {event.preventDefault(); event.stopPropagation(); toggle(false);}
    if (event.key==='Tab' && isMobile()) {
      const focusable=[...panel.querySelectorAll('button:not(:disabled),textarea,a[href],[tabindex="0"]')];
      const active=parentElement.activeElement || document.activeElement;
      if (event.shiftKey && active===focusable[0]) {event.preventDefault();focusable.at(-1).focus();}
      else if (!event.shiftKey && active===focusable.at(-1)) {event.preventDefault();focusable[0].focus();}
    }
  };
  const resize=()=>panel.setAttribute('aria-modal',String(isMobile()));
  document.addEventListener('keydown',onKey,true);
  window.addEventListener('resize',resize);
  applyOpen(ui.open,false);
  return ()=>{document.removeEventListener('keydown',onKey,true);window.removeEventListener('resize',resize);};
}
