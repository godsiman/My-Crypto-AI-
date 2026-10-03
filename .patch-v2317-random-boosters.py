from pathlib import Path

p=Path('index.html')
s=p.read_text(encoding='utf-8')

if 'V23.17' in s:
    print('already patched')
    raise SystemExit(0)

if 'V23.16' not in s:
    raise SystemExit('expected V23.16')
s=s.replace('V23.16','V23.17',1)

marker='const OFFICIAL_RECENT = ['
if marker not in s:
    raise SystemExit('OFFICIAL_RECENT marker missing')

insert=r'''
// Complete Random Booster / Select catalog audit.
// IMPORTANT: old app MasterData stores recoloured reprints under the ORIGINAL model_name,
// so relying on model_name alone drops variants such as CX-08-04/-05/-06.
// This registry is the product-level source of truth for blind-pack membership and suffixes.
const RANDOM_BOOSTER_CATALOG = [
  {code:"BX-14",name:"Random Booster Vol.1",count:6,variants:[
    {n:"01",parts:{blade:"SHARKEDGE",ratchet:"3-60",bit:"LF"}},
    {n:"02",parts:{blade:"SHARKEDGE",ratchet:"4-80",bit:"N"}},
    {n:"03",parts:{blade:"DRANSWORD",ratchet:"3-80",bit:"B"}},
    {n:"04",parts:{blade:"HELLSCYTHE",ratchet:"4-80",bit:"LF"}},
    {n:"05",parts:{blade:"KNIGHTSHIELD",ratchet:"4-60",bit:"LF"}},
    {n:"06",parts:{blade:"WIZARDARROW",ratchet:"3-60",bit:"T"}}
  ]},
  {code:"BX-16",name:"Viper Tail Select",count:3,variants:[
    {n:"01",parts:{blade:"VIPERTAIL",ratchet:"5-80",bit:"O"}},
    {n:"02",parts:{blade:"VIPERTAIL",ratchet:"4-60",bit:"F"}},
    {n:"03",parts:{blade:"VIPERTAIL",ratchet:"3-80",bit:"HN"}}
  ]},
  {code:"BX-24",name:"Random Booster Vol.2",count:6,variants:[
    {n:"01",parts:{blade:"WYVERNGALE",ratchet:"5-80",bit:"GB"}},
    {n:"02",parts:{blade:"WYVERNGALE",ratchet:"3-60",bit:"T"}},
    {n:"03",parts:{blade:"KNIGHTLANCE",ratchet:"4-60",bit:"GB"}},
    {n:"04",parts:{blade:"VIPERTAIL",ratchet:"5-60",bit:"F"}},
    {n:"05",parts:{blade:"LEONCLAW",ratchet:"3-80",bit:"HN"}},
    {n:"06",parts:{blade:"WIZARDARROW",ratchet:"4-80",bit:"GB"}}
  ]},
  {code:"BX-27",name:"Sphinx Cowl Select",count:3,variants:[
    {n:"01",parts:{blade:"SPHINXCOWL",ratchet:"9-80",bit:"GN"}},
    {n:"02",parts:{blade:"SPHINXCOWL",ratchet:"4-80",bit:"HT"}},
    {n:"03",parts:{blade:"SPHINXCOWL",ratchet:"5-60",bit:"O"}}
  ]},
  {code:"BX-31",name:"Random Booster Vol.3",count:6,variants:[
    {n:"01",parts:{blade:"TYRANNOBEAT",ratchet:"4-70",bit:"Q"}},
    {n:"02",parts:{blade:"TYRANNOBEAT",ratchet:"3-60",bit:"S"}},
    {n:"03",parts:{blade:"HELLSCHAIN",ratchet:"9-80",bit:"O"}},
    {n:"04",parts:{blade:"DRANDAGGER",ratchet:"4-70",bit:"P"}},
    {n:"05",parts:{blade:"SHARKEDGE",ratchet:"1-60",bit:"Q"}},
    {n:"06",parts:{blade:"RHINOHORN",ratchet:"5-80",bit:"Q"}}
  ]},
  {code:"UX-05",name:"Shinobi Shadow Select",count:3,variants:[
    {n:"01",parts:{blade:"SHINOBISHADOW",ratchet:"1-80",bit:"MN"}},
    {n:"02",parts:{blade:"SHINOBISHADOW",ratchet:"9-60",bit:"LF"}},
    {n:"03",parts:{blade:"SHINOBISHADOW",ratchet:"3-70",bit:"GP"}}
  ]},
  {code:"BX-35",name:"Random Booster Vol.4",count:6,variants:[
    {n:"01",parts:{blade:"BLACKSHELL",ratchet:"4-60",bit:"D"}},
    {n:"02",parts:{blade:"BLACKSHELL",ratchet:"9-80",bit:"B"}},
    {n:"03",parts:{blade:"UNICORNSTING",ratchet:"3-70",bit:"D"}},
    {n:"04",parts:{blade:"WIZARDROD",ratchet:"1-60",bit:"R"}},
    {n:"05",parts:{blade:"PHOENIXWING",ratchet:"5-80",bit:"H"}},
    {n:"06",parts:{blade:"VIPERTAIL",ratchet:"5-70",bit:"D"}}
  ]},
  {code:"BX-00",name:"Lightning L-Drago Random Booster",count:2,selector:"ldrago",variants:[
    {n:"01",label:"Lightning L-Drago 1-60F（Upper Type）",ratchet:"1-60",bit:"F",hint:"upper"},
    {n:"02",label:"Lightning L-Drago 1-60F（Rapid-Hit Type）",ratchet:"1-60",bit:"F",hint:"rapid"}
  ]},
  {code:"BX-36",name:"Whale Wave Select",count:3,variants:[
    {n:"01",parts:{blade:"WHALEWAVE",ratchet:"5-80",bit:"E"}},
    {n:"02",parts:{blade:"WHALEWAVE",ratchet:"4-70",bit:"HN"}},
    {n:"03",parts:{blade:"WHALEWAVE",ratchet:"3-80",bit:"GB"}}
  ]},
  {code:"UX-12",name:"Random Booster Vol.5",count:6,variants:[
    {n:"01",parts:{blade:"GHOSTCIRCLE",ratchet:"0-80",bit:"GB"}},
    {n:"02",parts:{blade:"GHOSTCIRCLE",ratchet:"4-60",bit:"H"}},
    {n:"03",parts:{blade:"SHINOBISHADOW",ratchet:"3-80",bit:"F"}},
    {n:"04",parts:{blade:"LEONCLAW",ratchet:"0-80",bit:"E"}},
    {n:"05",parts:{blade:"PHOENIXFEATHER",ratchet:"2-60",bit:"N"}},
    {n:"06",parts:{blade:"WYVERNGALE",ratchet:"0-80",bit:"C"}}
  ]},
  {code:"BX-39",name:"Shelter Drake Select",count:3,variants:[
    {n:"01",parts:{blade:"SHELTERDRAKE",ratchet:"7-80",bit:"GP"}},
    {n:"02",parts:{blade:"SHELTERDRAKE",ratchet:"5-70",bit:"O"}},
    {n:"03",parts:{blade:"SHELTERDRAKE",ratchet:"3-60",bit:"D"}}
  ]},
  {code:"CX-05",name:"Random Booster Vol.6",count:6,variants:[
    {n:"01",ratchet:"4-70",bit:"K",hint:"reaper"},
    {n:"02",ratchet:"4-55",bit:"D",hint:"reaper"},
    {n:"03",ratchet:"3-85",bit:"O",hint:"arc"},
    {n:"04",parts:{blade:"LEONCREST",ratchet:"9-80",bit:"K"}},
    {n:"05",parts:{blade:"PHOENIXRUDDER",ratchet:"4-70",bit:"LF"}},
    {n:"06",parts:{blade:"WHALEWAVE",ratchet:"7-60",bit:"K"}}
  ]},
  {code:"CX-06",name:"Fox Brush Select",count:3,variants:[
    {n:"01",ratchet:"9-70",bit:"GR",hint:"fox"},
    {n:"02",ratchet:"0-80",bit:"DB",hint:"fox"},
    {n:"03",ratchet:"2-60",bit:"U",hint:"fox"}
  ]},
  {code:"CX-08",name:"Random Booster Vol.7",count:6,variants:[
    {n:"01",label:"魔犬烈焰 W5-80WB",ratchet:"5-80",bit:"WB",hint:"cerberus"},
    {n:"02",label:"巨鯨烈焰 M3-85HT",ratchet:"3-85",bit:"HT",hint:"whale"},
    {n:"03",label:"魔犬幽冥 W1-60F",ratchet:"1-60",bit:"F",hint:"cerberus"},
    {n:"04",label:"蒼龍爆刃 5-80MN",parts:{blade:"DRANBUSTER",ratchet:"5-80",bit:"MN"}},
    {n:"05",label:"玄冥戰甲 7-70WB",parts:{blade:"BLACKSHELL",ratchet:"7-70",bit:"WB"}},
    {n:"06",label:"蒼穹龍騎士 4-55WB",parts:{blade:"COBALTDRAGOON",ratchet:"4-55",bit:"WB"},left:true}
  ]},
  {code:"UX-16",name:"Clock Mirage Select",count:3,sameCombo:true,variants:[
    {n:"01",parts:{blade:"CLOCKMIRAGE",ratchet:"9-65",bit:"B"},label:"時鐘幻象 9-65B（01）"},
    {n:"02",parts:{blade:"CLOCKMIRAGE",ratchet:"9-65",bit:"B"},label:"時鐘幻象 9-65B（02）"},
    {n:"03",parts:{blade:"CLOCKMIRAGE",ratchet:"9-65",bit:"B"},label:"時鐘幻象 9-65B（03）"}
  ]},
  {code:"UX-18",name:"Random Booster Vol.8",count:6,variants:[
    {n:"01",parts:{blade:"MUMMYCURSE",ratchet:"7-55",bit:"W"}},
    {n:"02",parts:{blade:"MUMMYCURSE",ratchet:"4-60",bit:"C"}},
    {n:"03",ratchet:"3-85",bit:"W",hint:"pegasus"},
    {n:"04",ratchet:"9-70",bit:"TP",hint:"sol"},
    {n:"05",parts:{blade:"DRANDAGGER",ratchet:"7-55",bit:"G"}},
    {n:"06",parts:{blade:"WEISSTIGER",ratchet:"4-80",bit:"LR"}}
  ]},
  {code:"BX-48",name:"Random Booster Vol.9",count:5,variants:[
    {n:"01",parts:{blade:"COBALTDRAGOON",ratchet:"9-80",bit:"F"},left:true},
    {n:"02",parts:{blade:"SHARKEDGE",ratchet:"4-70",bit:"E"}},
    {n:"03",parts:{blade:"MAMMOTHTUSK",ratchet:"7-60",bit:"S"}},
    {n:"04",parts:{blade:"HELLSCYTHE",ratchet:"3-85",bit:"GB"}},
    {n:"05",parts:{blade:"DRANBUSTER",ratchet:"2-80",bit:"Q"}}
  ]},
  {code:"CX-17",name:"Random Booster Vol.10",count:6,variants:[
    {n:"01",ratchet:"3-60",bit:"GU",hint:"unicorn"},
    {n:"02",ratchet:"1-80",bit:"GR",hint:"unicorn"},
    {n:"03",parts:{blade:"SAMURAISABER",ratchet:"9-65",bit:"LO"}},
    {n:"04",parts:{blade:"HELLSHAMMER",ratchet:"3-85",bit:"GU"}},
    {n:"05",parts:{blade:"TYRANNOBEAT",ratchet:"3-60",bit:"N"}},
    {n:"06",parts:{blade:"CRIMSONGARUDA",ratchet:"7-80",bit:"GU"}}
  ]},
  {code:"CX-18",name:"Brachio Whip Select",count:3,sameCombo:true,variants:[
    {n:"01",ratchet:"5-70",bit:"Nr",hint:"brachio",label:"Brachio Whip OW5-70Nr（01）"},
    {n:"02",ratchet:"5-70",bit:"Nr",hint:"brachio",label:"Brachio Whip OW5-70Nr（02）"},
    {n:"03",ratchet:"5-70",bit:"Nr",hint:"brachio",label:"Brachio Whip OW5-70Nr（03）"}
  ]},
  {code:"BX-50",name:"Random Booster Vol.11",count:6,variants:[
    {n:"01",parts:{blade:"HEAVENSRING",ratchet:"0-80",bit:"DS"}},
    {n:"02",parts:{blade:"HEAVENSRING",ratchet:"6-60",bit:"TP"}},
    {n:"03",parts:{blade:"IMPACTDRAKE",ratchet:"7-55",bit:"FB"}},
    {n:"04",parts:{blade:"GHOSTCIRCLE",ratchet:"M-85",bit:"DS"}},
    {n:"05",ratchet:"9-65",bit:"L",hint:"wolf"},
    {n:"06",ratchet:"0-80",bit:"WB",hint:"cerberus"}
  ]},
  {code:"CX-19",name:"Croco Tread Select",count:3,sameCombo:true,special:"SPECIAL:CX19",variants:[
    {n:"01",label:"巨鱷碾壓 TQ5-50GN（01）"},
    {n:"02",label:"巨鱷碾壓 TQ5-50GN（02）"},
    {n:"03",label:"巨鱷碾壓 TQ5-50GN（03）"}
  ]}
];

function randomBoosterPartKey(b,slot){ return b?.parts?.[slot]?.key || ""; }
function randomBoosterCandidateText(b){ return norm([b?.title,b?.combo,b?.model].filter(Boolean).join(" ")); }
function randomBoosterExistingMatch(candidates,v,used){
  const possible=candidates.filter(b=>{
    if(v.ratchet && randomBoosterPartKey(b,"ratchet")!==v.ratchet) return false;
    if(v.bit && randomBoosterPartKey(b,"bit")!==v.bit) return false;
    if(v.parts?.blade && randomBoosterPartKey(b,"blade")!==v.parts.blade) return false;
    if(v.hint && !randomBoosterCandidateText(b).includes(norm(v.hint))) return false;
    return true;
  });
  return possible.find(x=>!used.has(x.id)) || possible[0] || null;
}
function randomBoosterResolvedParts(v,existing,pack){
  if(existing?.parts && Object.values(existing.parts).some(Boolean)) return existing.parts;
  if(v.parts) return resolvePartMap(v.parts);
  if(pack.special){
    const sp=specialProductById(pack.special);
    if(sp) return resolvePartMap(sp.parts||{});
  }
  return {};
}
function randomBoosterComboLabel(parts,v){
  if(v.label) return v.label;
  const primary=parts.blade || parts.lock || parts.main || parts.metal || null;
  const name=primary ? displayNameForPart(primary) : "原廠配置";
  const r=parts.ratchet?.en || v.ratchet || "";
  const bit=parts.bit?.en || v.bit || "";
  return [name,r,bit].filter(Boolean).join(" ");
}
function applyRandomBoosterCatalog(){
  const original=state.stockBeys.slice();
  const managed=new Set(RANDOM_BOOSTER_CATALOG.map(p=>p.code));
  const keep=original.filter(b=>!managed.has(b.code));
  const rebuilt=[];

  for(const pack of RANDOM_BOOSTER_CATALOG){
    let candidates=original.filter(b=>b.code===pack.code);
    if(pack.selector==="ldrago") candidates=original.filter(b=>b.code==="BX-00" && /lightning|l-drago|エルドラゴ/i.test([b.title,b.combo,b.model].join(" ")));
    const used=new Set();
    for(const v of pack.variants){
      let existing=randomBoosterExistingMatch(candidates,v,used);
      if(existing) used.add(existing.id);
      // Same-combo Select products intentionally clone the same mechanical config into 01/02/03 colour variants.
      if(!existing && pack.sameCombo && candidates.length) existing=candidates[0];
      const parts=randomBoosterResolvedParts(v,existing,pack);
      const variantCode=`${pack.code}-${v.n}`;
      const combo=randomBoosterComboLabel(parts,v) || existing?.combo || v.label || variantCode;
      const line=pack.code.slice(0,2);
      rebuilt.push({
        ...(existing||{}),
        id:`RB:${variantCode}`,
        model:`RB:${variantCode}`,
        code:variantCode,
        baseCode:pack.code,
        variantCode,
        line,
        num:Number(pack.code.split("-")[1])||0,
        randomBooster:true,
        randomPack:pack.name,
        randomVariant:v.n,
        leftSpin:!!v.left,
        title:`${variantCode} ${combo}`,
        combo,
        parts,
        // Do not borrow a whole-Bey image from the original product/recolour.
        image_override:"",
        auditIncomplete:!Object.values(parts).some(Boolean)
      });
    }
  }
  state.stockBeys=[...keep,...rebuilt].sort((a,b)=>{
    const lo={BX:1,UX:2,CX:3};
    return ((lo[a.line]||9)-(lo[b.line]||9)) || (a.num-b.num) || String(a.code).localeCompare(String(b.code),undefined,{numeric:true});
  });

  const audit=[];
  for(const pack of RANDOM_BOOSTER_CATALOG){
    const got=state.stockBeys.filter(b=>b.randomBooster && b.baseCode===pack.code).length;
    if(got!==pack.count) audit.push(`${pack.code}:${got}/${pack.count}`);
  }
  if(audit.length) console.error("Random Booster catalog audit failed",audit);
  else console.info(`Random Booster catalog audit OK: ${RANDOM_BOOSTER_CATALOG.length} products / ${rebuilt.length} variants`);
}
'''
s=s.replace(marker,insert+'\n'+marker,1)

# Apply registry after MasterData stock build.
needle='''    ensureSpecialParts();\n    buildStockBeys();\n'''
repl='''    ensureSpecialParts();\n    buildStockBeys();\n    applyRandomBoosterCatalog();\n'''
if needle not in s:
    raise SystemExit('init buildStockBeys block missing')
s=s.replace(needle,repl,1)

# Random blind-pack variants must not borrow generic same-code colour images.
needle='''    if(!b || b.line==="CX" || b.parts[p.slot]?.key!==p.key) return "";'''
repl='''    if(!b || b.randomBooster || b.line==="CX" || b.parts[p.slot]?.key!==p.key) return "";'''
if needle not in s:
    raise SystemExit('stockVariantImage guard missing')
s=s.replace(needle,repl,1)

needle='''  function wholeImageForStock(b){\n    if(!b) return "";'''
repl='''  function wholeImageForStock(b){\n    if(!b) return "";\n    if(b.randomBooster) return b.image_override || "";'''
if needle not in s:
    raise SystemExit('wholeImageForStock marker missing')
s=s.replace(needle,repl,1)

# My-Bey image resolver: random variants stay strict/blank until exact colour image is mapped.
needle='''    if(!entry || entry.kind!=="stock") return "";'''
repl='''    if(!entry || entry.kind!=="stock") return "";\n    if(entry.randomBooster) return "";'''
if needle not in s:
    raise SystemExit('garageStockPartImage guard missing')
s=s.replace(needle,repl,1)

# Carry variant metadata into Garage entries.
needle='''      out.push({id:"stock:"+b.id,kind:"stock",ref:b.id,label:`${b.code}｜${b.combo}【${stockTypeLabel(b)}】`,code:b.code,title:b.combo,image:wholeImageForStock(b),parts:b.parts});'''
repl='''      out.push({id:"stock:"+b.id,kind:"stock",ref:b.id,stockId:b.id,label:`${b.code}｜${b.combo}【${stockTypeLabel(b)}】`,code:b.code,baseCode:b.baseCode||b.code,variantCode:b.variantCode||"",title:b.combo,image:wholeImageForStock(b),parts:b.parts,randomBooster:!!b.randomBooster,leftSpin:!!b.leftSpin});'''
if needle not in s:
    raise SystemExit('garage stock entry marker missing')
s=s.replace(needle,repl,1)

# CX-19 is now represented as three exact 01/02/03 variants in stock catalog; don't duplicate generic SPECIAL entry in pickers/Garage.
needle='''    for(const sp of SPECIAL_PRODUCTS){\n      out.push({id:"special:"+sp.id,kind:"special"'''
repl='''    for(const sp of SPECIAL_PRODUCTS.filter(x=>x.id!=="SPECIAL:CX19")){\n      out.push({id:"special:"+sp.id,kind:"special"'''
if needle not in s:
    raise SystemExit('garage special loop marker missing')
s=s.replace(needle,repl,1)

needle='''const recentSingles=`<optgroup label="🔥 2026/8–10 新品 / 限定">${SPECIAL_PRODUCTS.map(s=>'''
repl='''const recentSingles=`<optgroup label="🔥 2026/8–10 新品 / 限定">${SPECIAL_PRODUCTS.filter(s=>s.id!=="SPECIAL:CX19").map(s=>'''
if needle not in s:
    raise SystemExit('recentSingles marker missing')
s=s.replace(needle,repl,1)

# Footer: state the blind-pack audit source/behavior.
needle='''CX/EVA 的細部分件名稱另以玩家資料庫交叉核對。'''
repl='''CX/EVA 的細部分件名稱另以玩家資料庫交叉核對。Random Booster／Select 已另建完整 01／02／03…子款目錄核對，避免換色復刻零件因舊 model_name 被漏掉。'''
if needle not in s:
    raise SystemExit('footer marker missing')
s=s.replace(needle,repl,1)

p.write_text(s,encoding='utf-8')
print('patched',len(s))
