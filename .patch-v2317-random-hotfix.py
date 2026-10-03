from pathlib import Path
p=Path('index.html')
s=p.read_text(encoding='utf-8')

old='''  const managed=new Set(RANDOM_BOOSTER_CATALOG.map(p=>p.code));\n  const keep=original.filter(b=>!managed.has(b.code));'''
new='''  const managed=new Set(RANDOM_BOOSTER_CATALOG.map(p=>p.code));\n  const keep=original.filter(b=>{\n    if(!managed.has(b.code)) return true;\n    // BX-00 is reused by many limited/collaboration products. Only replace the\n    // Lightning L-Drago random-booster records; never wipe unrelated BX-00 items.\n    if(b.code==="BX-00") return !/lightning|l-drago|エルドラゴ/i.test([b.title,b.combo,b.model].join(" "));\n    return false;\n  });'''
if old not in s: raise SystemExit('managed keep block missing')
s=s.replace(old,new,1)

old='''  function stockSlots(b){\n    if(!b) return [];\n    if(b.line==="CX"){\n      return ["lock","main","assist","metal","over","ratchet","bit"].filter(s=>b.parts[s] || ["lock","main","assist","ratchet","bit"].includes(s));\n    }\n    return ["blade","ratchet","bit"];\n  }\n\n  function stockDisplayOption(b){\n    const stockParts=stockSlots(b).filter(s=>b.parts[s]).map(s=>b.parts[s]?.en).filter(Boolean);\n    const core=b.line==="CX"\n      ? [b.parts.lock?.zh||b.parts.lock?.en,b.parts.main?.zh||b.parts.main?.en,b.parts.assist?.en,b.parts.ratchet?.en,b.parts.bit?.en].filter(Boolean).join(" / ")\n      : [b.parts.blade?.zh||b.parts.blade?.en,b.parts.ratchet?.en,b.parts.bit?.en].filter(Boolean).join(" ");\n    return `${b.code}｜${core}【${stockTypeLabel(b)}】`;\n  }'''
new='''  function stockUsesCXStructure(b){\n    if(!b) return false;\n    return !!(b.parts?.lock || b.parts?.main || b.parts?.assist || b.parts?.metal || b.parts?.over);\n  }\n\n  function stockSlots(b){\n    if(!b) return [];\n    if(stockUsesCXStructure(b)){\n      return ["lock","main","assist","metal","over","ratchet","bit"].filter(s=>b.parts[s] || ["lock","main","assist","ratchet","bit"].includes(s));\n    }\n    return ["blade","ratchet","bit"].filter(s=>b.parts[s]);\n  }\n\n  function stockDisplayOption(b){\n    const cxStructure=stockUsesCXStructure(b);\n    const core=cxStructure\n      ? [b.parts.lock?.zh||b.parts.lock?.en,b.parts.main?.zh||b.parts.main?.en,b.parts.assist?.en,b.parts.ratchet?.en,b.parts.bit?.en].filter(Boolean).join(" / ")\n      : [b.parts.blade?.zh||b.parts.blade?.en,b.parts.ratchet?.en,b.parts.bit?.en].filter(Boolean).join(" ");\n    return `${b.code}｜${core || b.combo}【${stockTypeLabel(b)}】`;\n  }'''
if old not in s: raise SystemExit('stockSlots block missing')
s=s.replace(old,new,1)

old='''    setMode(b.line==="CX"?"cx":"classic");'''
new='''    setMode(stockUsesCXStructure(b)?"cx":"classic");'''
count=s.count(old)
if count < 2: raise SystemExit(f'expected >=2 stock mode occurrences, got {count}')
# First relevant occurrence is loadStockBey; second is resetToStock. Replace all exact b-based cases safely.
s=s.replace(old,new)

old='''    state.slot=b.line==="CX"?"lock":"blade";'''
new='''    state.slot=stockUsesCXStructure(b)?"lock":"blade";'''
if old not in s: raise SystemExit('stock slot mode marker missing')
s=s.replace(old,new,1)

p.write_text(s,encoding='utf-8')
print('hotfixed')
