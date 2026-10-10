"""Static review shell; disk-backed mutations are supplied by the local server."""

from __future__ import annotations

import html
import json
from pathlib import Path

from mouse_behavior.evaluation.manual_review import LABELS, review_catalog


def render_review_page(output_dir, rows, *, catalog=None, candidate_sha256=None,
                       supplemental_sha256=None, filename='index.html',
                       together_review_available=False,
                       approach_contact_review_available=False,
                       huddling_recheck_review_available=False,
                       contact_reclassification_review_available=False,
                       approach_no_contact_review_available=False,
                       approach_no_contact_recrop_review_available=False,
                       page_title=None, page_intro=None, page_rules=None):
    catalog = review_catalog(rows) if catalog is None else catalog
    source_video_label = (
        '原始样本片段' if filename in {
            'approach_no_contact_recrop_review.html', 'huddling_recheck_review.html'
        } else '完整原片'
    )
    cards = []
    options = '<option value="">请选择行为</option>' + ''.join(
        f'<option>{html.escape(label)}</option>' for label in LABELS)
    for card in catalog:
        def esc(value):
            return html.escape(str(value), quote=True)
        target_ids = ','.join(map(str, card['mouse_ids']))
        source_track_path = esc(f'/api/source-tracks/{card["source_key"]}.json')
        media = (
            f'<div><p>{source_video_label}（目标视觉 ID 框）</p><div class="review-video-frame">'
            f'<video class="original" controls preload="none" data-track-path="{source_track_path}" '
            f'data-target-ids="{esc(target_ids)}" data-fps="{card["fps"]:.8g}" '
            f'src="review_sources/{esc(card["source_key"])}.mp4"></video>'
            '<canvas class="tracking-overlay" aria-hidden="true"></canvas>'
            '<p class="overlay-status" aria-live="polite"></p></div></div>'
        )
        if card['export_path']:
            poster = f' poster="{esc(card["poster_path"])}"' if card['poster_path'] else ''
            if card.get('candidate_status') == 'rejected_locomotion_rearing_review_candidate':
                label = '站立攀爬待审核候选裁剪'
            elif card.get('candidate_status') == 'approach_contact_to_approach_review':
                label = '接近退回预览（重新审核接近过程）'
            elif card.get('candidate_status') == 'approach_no_contact_recut_review_candidate':
                label = '无身体接触的接近重裁（待人工复核）'
            elif card.get('candidate_status') == 'approach_no_contact_recrop_v1_review_candidate':
                label = '接近无接触裁剪片段（目标 ID 框已绘制，待人工审核）'
            elif card.get('candidate_status') == 'approach_no_contact_recrop_v2_review_candidate':
                label = '接近无接触裁剪片段（动态显示目标 ID 框，待人工审核）'
            elif card.get('candidate_status') == 'social_clustering_huddling_recheck':
                label = '社会聚集退回的扎堆复核候选'
            elif card.get('candidate_status') == 'together_contact_geometry_prescreen':
                label = '鼻头/鼻尾关键点初筛预览（待人工确认）'
            else:
                label = '候选裁剪 / 已通过片段'
            target_ids = ','.join(map(str, card['mouse_ids']))
            review_id = esc(card['review_id'])
            track_path = esc(f'{card["export_path"]}/tracks.json')
            baked_tracking = (
                card.get('candidate_status') == 'approach_no_contact_recrop_v1_review_candidate'
            )
            tracking_attributes = '' if baked_tracking else f' data-track-path="{track_path}"'
            video_attributes = (
                f'class="candidate" controls preload="none"{poster}{tracking_attributes} '
                f'data-target-ids="{esc(target_ids)}" data-fps="{card["fps"]:.8g}" '
                f'src="{esc(card["export_path"])}/clip.mp4"'
            )
            tracking_markup = '' if baked_tracking else (
                '<canvas class="tracking-overlay" aria-hidden="true"></canvas>'
                '<p class="overlay-status" aria-live="polite"></p>'
            )
            candidate_video = (
                f'<p class="candidate-identity"><strong>候选审核 ID：</strong><code>{review_id}</code> '
                f'· <strong>视觉轨迹 ID：</strong><code>{esc(target_ids)}</code></p>'
                f'<div class="review-video-frame"><div class="candidate-id-ribbon">'
                f'视觉轨迹 ID {esc(target_ids)} · 审核 ID {review_id}</div>'
                f'<video {video_attributes}></video>{tracking_markup}</div>'
            )
            media += f'<div><p>{label}</p>{candidate_video}</div>'
        else:
            media += '<div><p>暂无合格自动片段，可根据左侧原片填写行为、目标 ID 与帧范围。</p><video class="candidate" controls preload="none" hidden></video></div>'
        revision_notice = ''
        if card.get('candidate_update_message'):
            revision_notice = f'<p class="candidate-warning">{esc(card["candidate_update_message"])}</p>'
            if card['export_path']:
                revision_notice += f'<p>修正候选：原片帧 [{card["start_frame"]},{card["end_frame"]})。已保存的人工结论保留。</p><video class="revised-preview" controls preload="none" src="{esc(card["export_path"])}/clip.mp4"></video><button class="use-revised">采用修正范围（填入后再确认同步）</button>'
        candidate_reason_text = card.get('candidate_reason')
        if (card['behavior'] == 'Static'
                and candidate_reason_text == 'body_static_head_turn_allowed'):
            candidate_reason_text = (
                '身体中心活动范围半径不超过1.3375 cm；允许不超过0.2秒、位移不超过1.3375 cm的短暂挪动，'
                '也允许头部左右转动和有限抬头探索。若鼻尾投影长度降到参考体长的78%以下，且邻近2秒内恢复到1.25倍伸展，'
                '请转站立攀爬人工复核；顶视二维关键点不能测量真实抬头高度。'
            )
        candidate_reason = (f'<p>候选说明：{esc(candidate_reason_text)}</p>'
                            if candidate_reason_text else '')
        display_behavior = card.get('review_display_behavior', card['behavior'])
        cards.append(f'''<article data-id="{card['review_id']}" data-label="{esc(card['behavior'])}">
<h3><span class="behavior-label">{esc(display_behavior)}</span> · <span class="decision-status">待审核</span></h3><p>{esc(card['sample_id'])}</p>
<p>{source_video_label} {card['frame_count']} 帧 · {card['fps']:.5f} FPS · 候选来源 {esc(card['candidate_status'])}</p>{candidate_reason}
<div class="media">{media}</div>{revision_notice}<div class="editor">
<label>实际行为 <select class="behavior">{options}</select></label>
<label>目标视觉 ID <input class="ids" value="{esc(','.join(map(str,card['mouse_ids'])))}" placeholder="例如 3,6"></label>
<span class="available-ids">可用 ID：读取中…</span><br>
<label>起点帧（包含）<input class="start" type="number" min="0" max="{card['frame_count']-1}" value="{card['start_frame']}"></label>
<label>终点帧（不包含）<input class="end" type="number" min="1" max="{card['frame_count']}" value="{card['end_frame']}"></label>
<button class="seek">跳到起点</button><button class="set-start">当前画面设为起点</button><button class="set-end">当前画面设为末帧</button>
<label>播放速度 <select class="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label><br>
<label>备注 <input class="note" maxlength="2000" placeholder="行为依据或修改说明"></label></div>
<div class="actions"><button data-decision="approved" disabled>确认通过并同步</button><button data-decision="rejected" disabled>不通过</button><button data-decision="uncertain" disabled>暂不确定</button><button data-decision="pending" disabled>撤回 / 重置为待审核</button></div>
<p class="save-result" aria-live="polite"></p></article>''')
    data = json.dumps(catalog, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>行为片段人工确认</title>
<style>body{font:16px/1.5 system-ui;background:#f4f6f8;margin:24px;color:#172132}article{background:white;padding:18px;margin:18px 0;border-radius:12px;border:1px solid #d9e0e9}video{width:100%;max-height:420px;background:#111}.media{display:flex;gap:18px}.media>div{width:50%}p{overflow-wrap:anywhere}.candidate-identity{font-size:14px}.candidate-identity code{overflow-wrap:anywhere}.review-video-frame{position:relative;width:100%;aspect-ratio:16/9;max-height:420px;background:#111;overflow:hidden}.review-video-frame video{position:absolute;inset:0;width:100%;height:100%;max-height:none;object-fit:contain}.tracking-overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.candidate-id-ribbon{position:absolute;z-index:2;top:6px;left:6px;right:6px;padding:4px 7px;color:#fff;background:#101820d9;font:600 12px/1.35 system-ui;overflow-wrap:anywhere;pointer-events:none}.overlay-status{position:absolute;z-index:2;left:6px;bottom:2px;color:#ffe7b0;background:#101820c9;font-size:12px;margin:0}.pagination-bottom{display:flex;justify-content:center;align-items:center;gap:8px;margin:24px 0 48px}button,select,input{font:inherit;padding:7px;margin:4px}input{max-width:180px}input.note{max-width:none;width:min(650px,70vw)}header{position:sticky;top:0;background:#f4f6f8;padding:8px;z-index:1}.error{color:#a31e2e}.success{color:#177046}.available-ids{font-size:14px}.actions button[data-decision=approved]{background:#176545;color:white;border:1px solid #176545;border-radius:4px}button:disabled{opacity:.5}article[hidden]{display:none}@media(max-width:800px){.media{display:block}.media>div{width:100%}body{margin:12px}}</style></head><body>
<h1 id="page-title">行为片段人工确认</h1><p id="page-intro">逐段检查实际行为、目标视觉 ID 和裁剪范围。确认通过后，片段进入新数据集，同时同步原视频、轨迹、标注和 metadata；完整原片先备份。未通过或待审核片段不会进入新数据集。</p>SUPPLEMENTAL_LINK
<p id="global-rules">同一原片多个片段通过时拆成独立样本，修改标签同步对应行为目录。撤回确认会移出新数据集，并重建该原片的其他已通过样本；全部撤回后恢复完整原片。接近截止为约 3.5 cm（±0.5 cm）；追逐允许鼻尾接触。理毛与站立请按实际头部动作复核。</p>
<header><p id="scope-note" hidden></p><div id="connection" aria-live="polite">正在连接保存服务…</div><button id="retry">重新连接 / 刷新审核状态</button><button id="recover" hidden>关闭占用文件的程序后，恢复未完成保存</button><div id="summary">所有片段待审核</div><select id="label"><option value="">全部行为</option>FILTERS</select><select id="decision-filter"><option value="">全部审核状态</option><option value="pending">待审核</option><option value="approved">已通过</option><option value="rejected">未通过</option><option value="uncertain">暂不确定</option></select><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span><p id="dataset-path"></p><a href="/api/dataset-manifest" target="_blank">查看新数据集清单</a></header>
CARDS<nav class="pagination-bottom" aria-label="审核分页"><button id="prev-bottom" type="button">上一页</button><button id="next-bottom" type="button">下一页</button><span id="count-bottom" aria-live="polite"></span></nav><script id="catalog" type="application/json">CATALOG</script><script>
const candidateSha=CANDIDATE_SHA, supplementalSha=SUPPLEMENTAL_SHA;
const pageTitleOverride=PAGE_TITLE_OVERRIDE, pageIntroOverride=PAGE_INTRO_OVERRIDE, pageRulesOverride=PAGE_RULES_OVERRIDE;
const catalog=JSON.parse(document.getElementById('catalog').textContent), byId=new Map(catalog.map(c=>[c.review_id,c]));
const requestedBehavior=new URLSearchParams(location.search).get('behavior');
const behaviorAliases={'移动':'Locomotion','运动':'Locomotion','站立攀爬':'Rearing','静止':'Static','一起':'Together','接触行为复核':'ContactReview'};
const behaviorDisplayNames={'Approach':'接近（Approach）','ContactReview':'鼻头/鼻尾接触关键点初筛复核','Locomotion':'移动（Locomotion）','Rearing':'站立攀爬（Rearing）','Static':'静止（Static）'};
const scopedBehavior=behaviorAliases[requestedBehavior]||requestedBehavior;
const scopedCards=requestedBehavior?(scopedBehavior==='ContactReview'?catalog:catalog.filter(c=>c.behavior===scopedBehavior)):catalog;
const scopedIds=new Set(scopedCards.map(c=>c.review_id));
const trackingData=new WeakMap();
const MAX_BOX_INTERPOLATION_GAP=3;
function frameBox(frame,id){const detections=frame&&Array.isArray(frame.detections)?frame.detections:[];for(const detection of detections){const box=detection.box;if(Number(detection.track_id)===id&&Array.isArray(box)&&box.length===4&&box.every(Number.isFinite))return box;}return null;}
function boxAt(tracks,id,index){const direct=frameBox(tracks[index],id);if(direct)return{box:direct,estimated:false};let before=index-1;while(before>=0&&index-before<=MAX_BOX_INTERPOLATION_GAP+1&&!frameBox(tracks[before],id))before--;let after=index+1;while(after<tracks.length&&after-index<=MAX_BOX_INTERPOLATION_GAP+1&&!frameBox(tracks[after],id))after++;if(before<0||after>=tracks.length||after-before-1>MAX_BOX_INTERPOLATION_GAP)return null;const left=frameBox(tracks[before],id),right=frameBox(tracks[after],id),ratio=(index-before)/(after-before);return{box:left.map((value,i)=>value+(right[i]-value)*ratio),estimated:true};}
async function loadTrackingData(video){const path=video.dataset.trackPath,cached=trackingData.get(video);if(cached&&cached.path===path)return;try{const response=await fetch(path,{cache:'no-store'});const type=response.headers.get('content-type')||'';if(!response.ok||!type.includes('application/json'))throw new Error('track data unavailable');const tracks=await response.json();if(!Array.isArray(tracks)||!tracks.length)throw new Error('track data has an invalid shape');if(video.dataset.trackPath!==path)return;trackingData.set(video,{path,tracks});drawTracking(video);}catch(error){const status=video.parentElement&&video.parentElement.querySelector('.overlay-status');if(status)status.textContent='轨迹标记暂不可用；视频和审核仍可使用。';}}
function drawTracking(video){const wrapper=video.parentElement,canvas=wrapper&&wrapper.querySelector('.tracking-overlay');if(!canvas||!video.videoWidth||!video.videoHeight)return;const cached=trackingData.get(video);if(!cached||cached.path!==video.dataset.trackPath)return;const tracks=cached.tracks,frameIndex=Math.max(0,Math.min(tracks.length-1,Math.floor(video.currentTime*Number(video.dataset.fps)))),ids=[...new Set((video.dataset.targetIds||'').split(',').filter(Boolean).map(Number))],rect=wrapper.getBoundingClientRect(),dpr=window.devicePixelRatio||1;if(rect.width<=0||rect.height<=0)return;const width=Math.round(rect.width*dpr),height=Math.round(rect.height*dpr);if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);const scale=Math.min(rect.width/video.videoWidth,rect.height/video.videoHeight),offsetX=(rect.width-video.videoWidth*scale)/2,offsetY=(rect.height-video.videoHeight*scale)/2,palette=['#20ffd1','#ffcc3d','#ff7d8a','#8fa8ff'];ids.forEach((id,index)=>{const selected=boxAt(tracks,id,frameIndex);if(!selected)return;const box=selected.box,x=offsetX+box[0]*scale,y=offsetY+box[1]*scale,w=(box[2]-box[0])*scale,h=(box[3]-box[1])*scale;if(w<=0||h<=0)return;const color=palette[index%palette.length];ctx.lineWidth=3;ctx.strokeStyle=color;if(selected.estimated)ctx.setLineDash([8,5]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);const label=`visual ID ${id}${selected.estimated?'*':''}`;ctx.font='bold 14px system-ui';const textWidth=ctx.measureText(label).width+10,labelY=Math.max(offsetY,y-21);ctx.fillStyle='#073c34e8';ctx.fillRect(x,labelY,textWidth,20);ctx.fillStyle='#ffffff';ctx.fillText(label,x+5,labelY+15);});}
function trackLoop(video){drawTracking(video);if(!video.paused&&!video.ended)requestAnimationFrame(()=>trackLoop(video));}
document.querySelectorAll('video[data-track-path]').forEach(video=>{video.addEventListener('loadedmetadata',async()=>{await loadTrackingData(video);drawTracking(video);});video.addEventListener('timeupdate',()=>drawTracking(video));video.addEventListener('seeked',()=>drawTracking(video));video.addEventListener('play',async()=>{await loadTrackingData(video);trackLoop(video);});video.addEventListener('pause',()=>drawTracking(video));});
window.addEventListener('resize',()=>document.querySelectorAll('video[data-track-path]').forEach(drawTracking));
if(requestedBehavior){const display=behaviorDisplayNames[scopedBehavior]||scopedBehavior;document.title=display+'行为审核';document.getElementById('page-title').textContent=display+'行为审核';const intro=`本页固定审核${display}类别，只显示这类审核卡；逐段检查行为、视觉 ID 与裁剪范围，确认通过后仍按原流程同步新数据集和原始四类文件。`;const togetherRule='一起判定：允许两鼠有间隔且无互动，或挨在一起静止；排除接近、追逐、攻击和主动互动。每个确认片段必须至少1秒，保存时会校验时长。';const approachRule='接近接触退回复核：这些片段仍归接近审核，不预先改成一起。请核实接触之前是否存在符合接近定义的过程，并把裁剪终点放在身体接触发生之前；按接近规则核对起止距离和持续过程。';const contactRule='这是关键点几何初筛队列，建议行为仅作线索，不是确认标签。请逐段查看预览和完整原片，核对鼻头到对方头部或后端的实际接触类型，再选择鼻头接触、鼻尾接触或其他合适行为。';const staticRule='静止判定提示：持续至少1秒，身体中心活动范围半径不超过1.3375 cm；当前候选身体中心速度上限为13.375 cm/s。允许不超过0.2秒且位移不超过1.3375 cm的短暂挪动；头部左右转动与有限抬头探索本身不算移动。若鼻尾投影长度降到参考体长的78%以下，且邻近2秒内恢复到1.25倍伸展，需转站立攀爬人工复核。顶视二维关键点不能测真实抬头高度，单凭抬头不自动改判站立。';const scopedIntro=scopedBehavior==='Together'?intro+togetherRule:scopedBehavior==='Approach'?intro+approachRule:scopedBehavior==='ContactReview'?intro+contactRule:intro;document.getElementById('page-intro').textContent=scopedCards.length?(scopedBehavior==='Static'?intro+staticRule:scopedIntro):`没有找到“${requestedBehavior}”对应的审核卡，请检查类别名称。`;const note=document.getElementById('scope-note');note.hidden=false;note.textContent=scopedCards.length?`当前队列：${display}，共 ${scopedCards.length} 张审核卡。此页面不会切换到其他行为类别。`:`当前类别“${requestedBehavior}”没有审核卡。`;document.getElementById('label').hidden=true;document.getElementById('global-rules').hidden=true;}
if(pageTitleOverride){document.title=pageTitleOverride;document.getElementById('page-title').textContent=pageTitleOverride;}
if(pageIntroOverride)document.getElementById('page-intro').textContent=pageIntroOverride;
if(pageRulesOverride){const rules=document.getElementById('global-rules');rules.hidden=false;rules.textContent=pageRulesOverride;}
const cards=[...document.querySelectorAll('article')];let page=0,state={revision:0,decisions:{}},token='',connected=false,saving=false;
const names={pending:'待审核',approved:'已通过并同步',rejected:'未通过',uncertain:'暂不确定'}, detailCache=new Map();
function controls(){cards.forEach(c=>c.querySelectorAll('[data-decision]').forEach(b=>b.disabled=!connected||saving));}
async function request(url,options={}){const r=await fetch(url,{cache:'no-store',...options});const type=r.headers.get('content-type')||'';if(!type.includes('application/json'))throw new Error('保存服务未启动或返回了非 JSON；请重新连接。');const value=await r.json();if(!r.ok)throw new Error(value.error||`服务错误 ${r.status}`);return value;}
function isUnreviewedTransfer(base,saved){return !!base.review_origin_decision_updated_at&&!!base.review_origin_behavior&&saved.decision==='rejected'&&saved.behavior===base.review_origin_behavior&&saved.updated_at===base.review_origin_decision_updated_at;}
function updateSummary(){let n=state.counts||{},total=catalog.length;if(requestedBehavior){n={approved:0,rejected:0,uncertain:0,pending:0};for(const card of scopedCards){const saved=state.decisions[card.review_id]||{},carried=isUnreviewedTransfer(card,saved),decision=carried?'pending':(saved.decision||'pending');if(Object.prototype.hasOwnProperty.call(n,decision))n[decision]++;}total=scopedCards.length;const display=behaviorDisplayNames[scopedBehavior]||scopedBehavior;document.getElementById('summary').textContent=`${display}：通过 ${n.approved} · 未通过 ${n.rejected} · 暂不确定 ${n.uncertain} · 待审核 ${n.pending} / 共 ${total} 项`;}else{document.getElementById('summary').textContent=`通过 ${n.approved||0} · 未通过 ${n.rejected||0} · 暂不确定 ${n.uncertain||0} · 待审核 ${n.pending??catalog.length} / 共 ${catalog.length} 项`;}document.getElementById('dataset-path').textContent=state.confirmed_dataset?'新数据集：'+state.confirmed_dataset:'';}
async function details(card){const key=byId.get(card.dataset.id).source_key;if(!detailCache.has(key))detailCache.set(key,request('/api/source-details?source_key='+key).catch(e=>{detailCache.delete(key);throw e;}));try{const d=await detailCache.get(key);card.querySelector('.available-ids').textContent='可用 ID：'+d.mouse_ids.join(', ');}catch(e){card.querySelector('.available-ids').textContent='可用 ID 暂不可读：'+e.message;}}
function show(){let n=0;const label=document.getElementById('label').value,decision=document.getElementById('decision-filter').value;cards.forEach(c=>{const id=c.dataset.id,base=byId.get(id),d=state.decisions[id]||{decision:'pending'},carried=isUnreviewedTransfer(base,d),behavior=carried?(base.review_default_behavior||base.behavior):(d.behavior||base.behavior),effectiveDecision=carried?'pending':(d.decision||'pending');const match=scopedIds.has(id)&&(requestedBehavior||!label||behavior===label)&&(!decision||effectiveDecision===decision);const visible=match&&n>=page*12&&n<(page+1)*12;c.hidden=!visible;if(!visible)c.querySelectorAll('video').forEach(v=>v.pause());else if(connected)details(c);if(match)n++;});const pages=Math.max(1,Math.ceil(n/12));if(page>=pages){page=pages-1;return show();}const pageText=`第 ${page+1} / ${pages} 页，共 ${n} 项`;for(const suffix of ['', '-bottom']){document.getElementById('count'+suffix).textContent=pageText;document.getElementById('next'+suffix).disabled=(page+1)*12>=n;document.getElementById('prev'+suffix).disabled=page===0;}}
function fill(){cards.forEach(c=>{const base=byId.get(c.dataset.id),d=state.decisions[c.dataset.id]||{...base,decision:'pending',note:''},carried=isUnreviewedTransfer(base,d),behavior=carried?(base.review_default_behavior||''):d.behavior;c.querySelector('.behavior').value=behavior;c.querySelector('.behavior-label').textContent=behavior||base.review_suggestion_label||base.behavior;c.querySelector('.ids').value=d.mouse_ids.join(',');c.querySelector('.start').value=d.start_frame;c.querySelector('.end').value=d.end_frame;c.querySelector('.note').value=d.note||'';c.querySelector('.decision-status').textContent=carried?(base.review_transfer_pending_label||'待重新审核（原结论待复核）'):names[d.decision];const source=c.querySelector('.original');if(source){source.dataset.targetIds=d.mouse_ids.join(',');drawTracking(source);}const v=c.querySelector('.candidate'),overlay=v.parentElement&&v.parentElement.querySelector('.tracking-overlay');if(d.decision==='approved'){if(overlay){overlay.hidden=false;v.dataset.trackPath='/api/confirmed/'+c.dataset.id+'/tracks.json?revision='+state.revision;v.dataset.targetIds=d.mouse_ids.join(',');v.dataset.confirmed='true';}v.hidden=false;v.removeAttribute('poster');v.src='/api/confirmed/'+c.dataset.id+'/clip.mp4?revision='+state.revision;}else if(base.export_path){if(overlay){overlay.hidden=false;v.dataset.trackPath=base.export_path+'/tracks.json';v.dataset.targetIds=d.mouse_ids.join(',');v.dataset.confirmed='false';}v.hidden=false;if(base.poster_path)v.setAttribute('poster',base.poster_path);v.src=base.export_path+'/clip.mp4';}else{v.hidden=true;v.removeAttribute('src');if(overlay)overlay.hidden=true;}});updateSummary();controls();show();}
async function connect(){try{state=await request('/api/state');if((state.candidate_sha256||null)!==candidateSha||(state.supplemental_sha256||null)!==supplementalSha)throw new Error('候选版本已更新，请重新加载整个页面后保存。');token=state.token;connected=!state.recovery_error;document.getElementById('recover').hidden=connected;document.getElementById('connection').className=connected?'success':'error';document.getElementById('connection').textContent=connected?'审核状态已从磁盘加载；通过后同步保存。':state.recovery_error;fill();}catch(e){connected=false;controls();document.getElementById('connection').className='error';document.getElementById('connection').textContent=e.message+' 播放和筛选仍可使用。';}}
function values(c,decision){const ids=c.querySelector('.ids').value.trim().split(/[，,\s]+/).filter(Boolean);if(!ids.every(i=>/^\d+$/.test(i)))throw new Error('目标 ID 请填写整数，用逗号分隔。');const start=c.querySelector('.start').value,end=c.querySelector('.end').value;if(!/^\d+$/.test(start)||!/^\d+$/.test(end))throw new Error('起点与终点必须是整数帧号。');return {review_id:c.dataset.id,decision,behavior:c.querySelector('.behavior').value,mouse_ids:ids.map(Number),start_frame:Number(start),end_frame:Number(end),note:c.querySelector('.note').value,revision:state.revision,candidate_sha256:candidateSha,supplemental_sha256:supplementalSha,request_id:crypto.randomUUID()};}
cards.forEach(c=>{const input=c.querySelector('.ids'),videos=[...c.querySelectorAll('video[data-track-path]')];if(!input||!videos.length)return;input.addEventListener('input',()=>{const ids=input.value.trim().split(/[，,\s]+/).filter(Boolean).join(',');videos.forEach(video=>{video.dataset.targetIds=ids;drawTracking(video);});});});
cards.forEach(c=>{const base=byId.get(c.dataset.id),original=c.querySelector('.original'),result=c.querySelector('.save-result');c.querySelector('.behavior').value=base.behavior;c.querySelector('.seek').onclick=()=>original.currentTime=Number(c.querySelector('.start').value)/base.fps;c.querySelector('.set-start').onclick=()=>c.querySelector('.start').value=Math.min(base.frame_count-1,Math.max(0,Math.floor(original.currentTime*base.fps)));c.querySelector('.set-end').onclick=()=>c.querySelector('.end').value=Math.min(base.frame_count,Math.max(1,Math.floor(original.currentTime*base.fps)+1));c.querySelector('.speed').onchange=e=>c.querySelectorAll('video').forEach(v=>v.playbackRate=Number(e.target.value));const revised=c.querySelector('.use-revised');if(revised)revised.onclick=()=>{c.querySelector('.start').value=base.start_frame;c.querySelector('.end').value=base.end_frame;c.querySelector('.behavior').value=base.behavior;c.querySelector('.ids').value=base.mouse_ids.join(',');original.currentTime=base.start_frame/base.fps;result.textContent='修正范围已填入，尚未保存；核对修正候选后点击确认通过并同步。';};c.querySelectorAll('[data-decision]').forEach(b=>b.onclick=async()=>{if(saving)return;let payload;try{payload=values(c,b.dataset.decision);}catch(e){result.className='save-result error';result.textContent=e.message;return;}saving=true;controls();result.className='save-result';result.textContent=payload.decision==='approved'?'正在备份、裁剪并同步文件…':'正在保存审核结论…';try{state=await request('/api/decision',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(payload)});token=state.token;result.className='save-result success';result.textContent=payload.decision==='approved'?'已通过：新数据集与原始四类文件已同步，完整原片备份已保留。':'已保存。';fill();}catch(e){result.className='save-result error';result.textContent=e.message+' 未收到成功回执，请刷新状态核对后再重试。';}finally{saving=false;controls();}});});
document.getElementById('label').onchange=document.getElementById('decision-filter').onchange=()=>{page=0;show();};function changePage(delta){const control=document.getElementById(delta>0?'next':'prev');if(control.disabled)return;page+=delta;show();window.scrollTo(0,0);}for(const suffix of ['', '-bottom']){document.getElementById('prev'+suffix).onclick=()=>changePage(-1);document.getElementById('next'+suffix).onclick=()=>changePage(1);}document.getElementById('retry').onclick=connect;document.getElementById('recover').onclick=async()=>{try{await request('/api/recover',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:'{}'});await connect();}catch(e){document.getElementById('connection').textContent=e.message;}};show();connect();
</script></body></html>'''
    filters = ''.join(f'<option>{html.escape(label)}</option>' for label in sorted(set(LABELS) | {c['behavior'] for c in catalog}))
    # Replace the catalog last, so user strings cannot become template substitutions.
    page = page.replace('FILTERS', filters, 1).replace('CARDS', ''.join(cards), 1)
    links = []
    if supplemental_sha256 and filename != 'rearing_review.html':
        links.append('<a href="/rearing_review.html?behavior=Rearing">打开站立攀爬专项审核页</a>')
    if together_review_available and filename != 'together_review.html':
        links.append('<a href="/together_review.html?behavior=Together">打开一起（Together）专项审核页</a>')
    if huddling_recheck_review_available and filename != 'huddling_recheck_review.html':
        links.append('<a href="/huddling_recheck_review.html?behavior=Huddling">打开社会聚集退回的扎堆复核页</a>')
    if approach_contact_review_available and filename != 'approach_contact_review.html':
        links.append('<a href="/approach_contact_review.html?behavior=Approach">打开接近接触退回复核页</a>')
    if approach_no_contact_review_available and filename != 'approach_no_contact_review.html':
        links.append('<a href="/approach_no_contact_review.html?behavior=Approach">打开接近无接触重裁复核页</a>')
    if (approach_no_contact_recrop_review_available
            and filename != 'approach_no_contact_recrop_review.html'):
        links.append('<a href="/approach_no_contact_recrop_review.html?behavior=Approach">打开接近无接触重裁审核页</a>')
    if (contact_reclassification_review_available
            and filename != 'contact_reclassification_review.html'):
        links.append('<a href="/contact_reclassification_review.html?behavior=ContactReview">打开鼻头/鼻尾接触关键点初筛复核页</a>')
    supplemental_link = '<p>' + ' · '.join(links) + '</p>' if links else ''
    page = page.replace('SUPPLEMENTAL_LINK', supplemental_link, 1)
    page = page.replace('PAGE_TITLE_OVERRIDE', json.dumps(page_title), 1)
    page = page.replace('PAGE_INTRO_OVERRIDE', json.dumps(page_intro), 1)
    page = page.replace('PAGE_RULES_OVERRIDE', json.dumps(page_rules), 1)
    page = page.replace('CANDIDATE_SHA', json.dumps(candidate_sha256), 1)
    page = page.replace('SUPPLEMENTAL_SHA', json.dumps(supplemental_sha256), 1)
    page = page.replace('<script id="catalog" type="application/json">CATALOG</script>',
                        '<script id="catalog" type="application/json">'+data+'</script>', 1)
    (Path(output_dir) / filename).write_text(page, encoding='utf-8', newline='\n')
