from pathlib import Path

p = Path('index.html')
s = p.read_text(encoding='utf-8')

if '>V23.15</span>' not in s:
    raise SystemExit('expected V23.15')
s = s.replace('>V23.15</span>', '>V23.16</span>', 1)

start = s.index('  function garageEntryPartImage(entry,slot,p){')
end = s.index('\n  function showGarageBeyDetail(entry){', start)
new_resolver = r'''  function garageStockPartImage(entry,slot){
    if(!entry || entry.kind!=="stock") return "";

    // A product code can contain multiple Random Booster / recolour variants.
    // Code-only filenames are safe only when MasterData has exactly one stock model
    // for that code. Otherwise leave the picture blank rather than show a wrong colour.
    const sameCode=state.stockBeys.filter(b=>b.code===entry.code);
    if(sameCode.length!==1) return "";
    const stock=sameCode[0];
    if(entry.stockId && stock.id!==entry.stockId) return "";

    // Standard BX/UX product pages use 01/02 = Blade, 03 = Ratchet, 04 = Bit.
    // CX has more upper components and no single universal suffix layout, so CX
    // requires an explicit product-specific mapping and otherwise stays blank.
    if(stock.line==="CX") return "";
    const base=entry.code.replace("-","");
    const suffixes=slot==="blade"
      ? ["_01@1","_01_list","_01","_02@1","_02_list","_02"]
      : slot==="ratchet"
      ? ["_03@1","_03_list","_03"]
      : slot==="bit"
      ? ["_04@1","_04_list","_04"]
      : [];
    for(const suffix of suffixes){
      const hit=allImageKey(base,suffix);
      if(hit) return hit;
    }
    return "";
  }

  function garageEntryPartImage(entry,slot,p){
    if(!entry || !p) return "";
    if(entry.partImages?.[slot]) return entry.partImages[slot];

    // BX-49 DranStrike 4-50FF: verified TAKARA TOMY product-specific part images.
    if(entry.code==="BX-49"){
      if(slot==="blade") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/BX49_01_list.png";
      if(slot==="ratchet") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/BX49_03_list.png";
      if(slot==="bit") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/BX49_04_list.png";
    }

    // UX-21 verified product-specific official part images.
    if(entry.kind==="setbey"){
      if(entry.beyId==="UX21-01" && slot==="blade") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/UX21_02%401.png";
      if(entry.beyId==="UX21-01" && slot==="bit") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/UX21_04%401.png";
      if(entry.beyId==="UX21-02" && slot==="blade") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/UX21_05%401.png";
      if(entry.beyId==="UX21-02" && slot==="bit") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/UX21_08%401.png";
      if(entry.beyId==="UX21-03" && slot==="blade") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/UX21_09%401.png";
      if(entry.beyId==="UX21-03" && slot==="bit") return "https://beyblade.takaratomy.co.jp/beyblade-x/lineup/_image/UX21_12%401.png";
    }

    // Normal BX/UX stock: only use a product-code image when that code resolves to
    // one exact stock model. Random Booster/recolour variants stay blank.
    const stockHit=garageStockPartImage(entry,slot);
    if(stockHit) return stockHit;

    // Strict for ALL My Bey originals. Never fall back to generic findImage(p),
    // because the same mechanical code can have a different product colourway.
    return "";
  }
'''
s = s[:start] + new_resolver + s[end:]

old_set = 'showGarageBeyDetail({kind:"setbey",code:entry.code,title:b.name,image:b.image,parts:resolvePartMap(b.parts||{blade:b.blade,ratchet:b.ratchet,bit:b.bit}),integrated:!!b.ratchetIntegrated});'
new_set = 'showGarageBeyDetail({kind:"setbey",beyId:b.id,code:entry.code,title:b.name,image:b.image,parts:resolvePartMap(b.parts||{blade:b.blade,ratchet:b.ratchet,bit:b.bit}),integrated:!!b.ratchetIntegrated,partImages:b.partImages||{},partLabels:b.partLabels||{},partVariantIds:b.partVariantIds||{}});'
if old_set not in s:
    raise SystemExit('setbey detail call not found')
s = s.replace(old_set, new_set, 1)

old_strict = 'const strictProductImages=entry.kind==="special" || entry.kind==="setbey" || entry.code==="BX-49";'
if old_strict not in s:
    raise SystemExit('strictProductImages line not found')
s = s.replace(old_strict, 'const strictProductImages=true;', 1)

old_desc = '${strictProductImages?"上方優先顯示這個商品自己的官方整顆／拆件素材；下方零件只有核對到商品專屬圖片才顯示，不會再拿別顆同代號零件的配色圖代替。":"點下面任一零件，可看零件實物圖、能力資料，以及還有哪些原廠陀螺也有這個零件。"}'
new_desc = '上方顯示這個商品自己的整顆素材；下方零件只有能對到這個商品／配色的分件圖才顯示。找不到專屬圖就留白，不會拿其他陀螺同代號零件圖代替。'
if old_desc not in s:
    raise SystemExit('detail description not found')
s = s.replace(old_desc, new_desc, 1)

old_note = '這裡只顯示這顆原廠配置。特殊／限定配色若沒有商品專屬零件圖，就留白標示，不再用其他商品的共用圖冒充。'
new_note = '這裡只顯示這顆原廠配置。BX／UX／CX 都採商品配色嚴格模式：沒有核對到該商品自己的分件圖就留白，不使用其他商品的共用配色圖。'
if old_note not in s:
    raise SystemExit('detail note not found')
s = s.replace(old_note, new_note, 1)

p.write_text(s, encoding='utf-8')
print('patched V23.16', len(s))
