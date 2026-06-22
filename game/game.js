/* =====================================================================
   CIVILIZATION 2026 — a turn-based 4X-lite strategy game for mobile.
   Vanilla JS, no dependencies. Designed for iPhone Safari / PWA.

   Lead a civilization from the STONE AGE all the way to the
   SINGULARITY across ten eras of history — discovering writing,
   currency, gunpowder, electricity, the Internet, AI and fusion —
   while surviving pandemics, financial crises, cyber-war and climate
   shocks. Choose a historical civilization or create your own.
   ===================================================================== */
'use strict';

/* ----------------------------- Config ----------------------------- */
const MAP_W = 7, MAP_H = 9;            // grid dimensions (portrait-friendly)
const WIN_TURN = 240;                  // score victory deadline
const GROW_BASE = 12;                  // food needed to grow (×pop)

const TERRAIN = {
  ocean:    {name:'Ocean',    cls:'t-ocean',    ico:'🌊', food:1, prod:0, gold:2, sci:0, land:false},
  coast:    {name:'Coast',    cls:'t-coast',    ico:'🐟', food:2, prod:0, gold:2, sci:0, land:false},
  grass:    {name:'Grassland',cls:'t-grass',    ico:'🌱', food:3, prod:1, gold:0, sci:0, land:true},
  plains:   {name:'Plains',   cls:'t-plains',   ico:'🌾', food:2, prod:2, gold:1, sci:0, land:true},
  forest:   {name:'Forest',   cls:'t-forest',   ico:'🌲', food:1, prod:3, gold:0, sci:0, land:true},
  hills:    {name:'Hills',    cls:'t-hills',    ico:'⛰️', food:1, prod:3, gold:1, sci:0, land:true},
  mountain: {name:'Mountains',cls:'t-mountain', ico:'🏔️', food:0, prod:2, gold:2, sci:1, land:true},
  desert:   {name:'Desert',   cls:'t-desert',   ico:'🏜️', food:0, prod:1, gold:3, sci:0, land:true},
  tundra:   {name:'Tundra',   cls:'t-tundra',   ico:'❄️', food:1, prod:1, gold:1, sci:0, land:true},
};

/* ----------------------------- Civilizations ----------------------------- */
// Each civ has a bonus bundle `b`. Bonuses:
//   foodMul/prodMul/goldMul/sciMul  – per-city yield multipliers
//   buildCostMul / settleMul        – cheaper buildings / settling (<1 = cheaper)
//   healthBonus / stabBonus         – raise the equilibrium of the meters
//   startGold                       – extra starting treasury
const CIVS = {
  nile:     {name:'Kingdom of the Nile', ico:'🏺', color:'#e0b84c',
             blurb:'Masters of the fertile river — cities flourish and grow.', b:{foodMul:1.25}},
  hellas:   {name:'Hellenic League',     ico:'🏛️', color:'#5bc8ff',
             blurb:'Cradle of philosophy, mathematics and science.', b:{sciMul:1.20}},
  imperium: {name:'The Imperium',        ico:'🦅', color:'#d05a5a',
             blurb:'Legendary engineers — build faster and stronger.', b:{prodMul:1.18, buildCostMul:0.9}},
  silk:     {name:'Silk Road Dynasty',   ico:'🐉', color:'#f0883e',
             blurb:'Wealthy merchants of the great trade roads.', b:{goldMul:1.22, startGold:20}},
  norse:    {name:'Norse Clans',         ico:'⚓', color:'#7fd0c0',
             blurb:'Bold explorers who settle new land cheaply.', b:{prodMul:1.08, settleMul:0.65}},
  republic: {name:'Maritime Republic',   ico:'⛵', color:'#9b8cff',
             blurb:'Banking and trade keep the economy resilient.', b:{goldMul:1.12, stabBonus:12}},
};
const DEFAULT_CIV = {name:'New Republic', ico:'🌍', color:'#3fb6ff', b:{}};

// Strengths offered when creating a custom civilization.
const FOCUSES = {
  agrarian:    {label:'🌾 Agrarian',     b:{foodMul:1.20}},
  industrious: {label:'🏭 Industrious',  b:{prodMul:1.20}},
  mercantile:  {label:'🪙 Mercantile',   b:{goldMul:1.20}},
  scholarly:   {label:'📚 Scholarly',    b:{sciMul:1.20}},
  expansionist:{label:'⛵ Expansionist', b:{settleMul:0.6}},
  resilient:   {label:'🛡️ Resilient',   b:{healthBonus:8, stabBonus:8}},
};

/* ----------------------------- Buildings ----------------------------- */
// city.* effects are per-city additive yields.
// nat.* flags (resist/bonus) are summed across every owned building.
const BUILDINGS = {
  granary:    {name:'Granary',        ico:'🌾', cost:12,  maint:0, tech:'pottery',
               city:{food:2}, desc:'+2 food. Stores the harvest.'},
  workshop:   {name:'Workshop',       ico:'🛠️', cost:16,  maint:1, tech:'bronze_working',
               city:{prod:2}, desc:'+2 production.'},
  library:    {name:'Library',        ico:'📚', cost:18,  maint:1, tech:'writing',
               city:{sci:2}, desc:'+2 science.'},
  market:     {name:'Market',         ico:'🪙', cost:20,  maint:1, tech:'currency',
               city:{gold:3}, desc:'+3 gold from trade.'},
  walls:      {name:'City Walls',     ico:'🧱', cost:18,  maint:1, tech:'masonry',
               city:{}, nat:{stab:5}, desc:'Protects the city. +stability.'},
  aqueduct:   {name:'Aqueduct',       ico:'🚰', cost:24,  maint:1, tech:'engineering',
               city:{food:2}, nat:{health:5}, desc:'+2 food, raises public health.'},
  university: {name:'University',      ico:'🎓', cost:32,  maint:2, tech:'education',
               city:{sci:4}, desc:'+4 science.'},
  bank:       {name:'Bank',           ico:'🏦', cost:36,  maint:2, tech:'banking',
               city:{gold:4}, nat:{stab:6, crisisResist:1}, desc:'+4 gold, stabilises the economy.'},
  harbor:     {name:'Harbor',         ico:'⚓', cost:30,  maint:1, tech:'navigation',
               city:{food:2, gold:2}, desc:'+2 food, +2 gold from the sea.'},
  manufactory:{name:'Manufactory',    ico:'🏭', cost:46,  maint:2, tech:'industrialization',
               city:{prod:5}, desc:'+5 production. The engine of industry.'},
  powerplant: {name:'Power Plant',    ico:'⚡', cost:52,  maint:3, tech:'electricity',
               city:{prod:3, gold:1}, desc:'+3 prod, +1 gold.'},
  hospital:   {name:'Hospital',       ico:'🏥', cost:55,  maint:2, tech:'medicine',
               city:{}, nat:{health:8, pandemicResist:1}, desc:'Raises public health, resists pandemics.'},
  datacenter: {name:'Data Center',    ico:'🖥️', cost:60,  maint:3, tech:'computing',
               city:{sci:5, gold:1}, desc:'+5 science, +1 gold.'},
  fiber:      {name:'Fiber & 5G',     ico:'🛰️', cost:60,  maint:2, tech:'internet',
               city:{gold:3, sci:2}, desc:'+3 gold, +2 science. Connects the city.'},
  exchange:   {name:'Stock Exchange', ico:'💹', cost:66,  maint:2, tech:'fintech',
               city:{gold:6}, nat:{stab:6, crisisResist:1}, desc:'+6 gold, calms markets.'},
  solar:      {name:'Renewable Grid', ico:'🔆', cost:60,  maint:2, tech:'renewables',
               city:{prod:2, gold:1}, nat:{climateResist:2}, desc:'+2 prod, +1 gold, cuts climate risk.'},
  biolab:     {name:'Biotech Lab',    ico:'🧬', cost:72,  maint:3, tech:'biotech',
               city:{sci:3}, nat:{health:8, pandemicResist:2}, desc:'+3 science, strong pandemic defence.'},
  ailab:      {name:'AI Lab',         ico:'🤖', cost:82,  maint:4, tech:'machine_learning',
               city:{sci:8, prod:2}, desc:'+8 science, +2 production. Compounds research.'},
  cyber:      {name:'Cyber Command',  ico:'🛡️', cost:72,  maint:3, tech:'cybersecurity',
               city:{}, nat:{stab:5, cyberResist:2}, desc:'Defends against cyber-attacks.'},
  quantum:    {name:'Quantum Hub',    ico:'⚛️', cost:100, maint:5, tech:'quantum',
               city:{sci:12}, desc:'+12 science. Quantum-accelerated R&D.'},
  fusion:     {name:'Fusion Plant',   ico:'☢️', cost:130, maint:4, tech:'fusion',
               city:{prod:9, gold:3}, nat:{climateResist:4}, desc:'+9 prod, +3 gold, clean limitless power.'},
};

/* ----------------------------- Tech tree ----------------------------- */
const ERAS = ['Stone Age','Bronze Age','Classical Age','Medieval Age','Renaissance',
              'Industrial Age','Modern Age','Information Age','AI Age','Singularity Age'];

const TECHS = {
  // ---- Stone Age (0) ----
  agriculture:  {name:'Agriculture',     ico:'🌱', era:0, cost:18,  req:[], note:'+1 food per city.', flat:{food:1}},
  pottery:      {name:'Pottery',         ico:'🏺', era:0, cost:20,  req:[], unlock:['granary'], note:'Granaries store food.'},
  mining:       {name:'Mining',          ico:'⛏️', era:0, cost:24,  req:[], note:'+1 production per city.', flat:{prod:1}},
  writing:      {name:'Writing',         ico:'✍️', era:0, cost:28,  req:['pottery'], unlock:['library'], note:'Libraries. +1 science/city.', flat:{sci:1}},
  // ---- Bronze Age (1) ----
  bronze_working:{name:'Bronze Working', ico:'🔱', era:1, cost:38,  req:['mining'], unlock:['workshop'], note:'Workshops boost production.'},
  masonry:      {name:'Masonry',         ico:'🧱', era:1, cost:40,  req:['mining'], unlock:['walls'], note:'City Walls. Stability.'},
  the_wheel:    {name:'The Wheel',       ico:'🛞', era:1, cost:42,  req:['bronze_working'], note:'+1 gold per city.', flat:{gold:1}},
  currency:     {name:'Currency',        ico:'🪙', era:1, cost:46,  req:['writing'], unlock:['market'], note:'Markets enable trade.'},
  // ---- Classical Age (2) ----
  mathematics:  {name:'Mathematics',     ico:'📐', era:2, cost:62,  req:['currency'], note:'+1 science per city.', flat:{sci:1}},
  iron_working: {name:'Iron Working',    ico:'⚒️', era:2, cost:66,  req:['bronze_working'], note:'+1 production per city.', flat:{prod:1}},
  engineering:  {name:'Engineering',     ico:'🏗️', era:2, cost:70,  req:['masonry','the_wheel'], unlock:['aqueduct'], note:'Aqueducts. Food & health.'},
  philosophy:   {name:'Philosophy',      ico:'🧠', era:2, cost:72,  req:['writing','mathematics'], note:'+health & +stability.', nat:{health:6, stab:6}},
  // ---- Medieval Age (3) ----
  feudalism:    {name:'Feudalism',       ico:'🛡️', era:3, cost:92,  req:['iron_working'], note:'+stability.', nat:{stab:8}},
  education:    {name:'Education',        ico:'🎓', era:3, cost:104, req:['philosophy'], unlock:['university'], note:'Universities. Big science.'},
  machinery:    {name:'Machinery',       ico:'⚙️', era:3, cost:104, req:['engineering','iron_working'], note:'+1 production per city.', flat:{prod:1}},
  banking:      {name:'Banking',         ico:'🏦', era:3, cost:116, req:['currency','feudalism'], unlock:['bank'], note:'Banks. Wealth & stability.'},
  // ---- Renaissance (4) ----
  printing_press:{name:'Printing Press', ico:'🖨️', era:4, cost:138, req:['education'], note:'+2 science per city.', flat:{sci:2}},
  gunpowder:    {name:'Gunpowder',       ico:'💥', era:4, cost:146, req:['machinery'], note:'+stability.', nat:{stab:6}},
  navigation:   {name:'Navigation',      ico:'🧭', era:4, cost:148, req:['mathematics'], unlock:['harbor'], note:'Harbors. Food & gold from the sea.'},
  astronomy:    {name:'Astronomy',       ico:'🔭', era:4, cost:158, req:['printing_press'], note:'+1 science per city.', flat:{sci:1}},
  // ---- Industrial Age (5) ----
  industrialization:{name:'Industrialization', ico:'🏭', era:5, cost:196, req:['banking','machinery'], unlock:['manufactory'], note:'Manufactories. +1 prod/city.', flat:{prod:1}},
  electricity:  {name:'Electricity',     ico:'⚡', era:5, cost:214, req:['astronomy'], unlock:['powerplant'], note:'Power Plants.'},
  railroad:     {name:'Railroad',        ico:'🚂', era:5, cost:214, req:['industrialization'], note:'+2 gold per city.', flat:{gold:2}},
  sanitation:   {name:'Sanitation',      ico:'🚿', era:5, cost:206, req:['industrialization'], note:'+health everywhere.', nat:{health:8}},
  // ---- Modern Age (6) ----
  combustion:   {name:'Combustion',      ico:'🚗', era:6, cost:262, req:['railroad','electricity'], note:'+2 production per city.', flat:{prod:2}},
  medicine:     {name:'Medicine',        ico:'💊', era:6, cost:262, req:['sanitation'], unlock:['hospital'], note:'Hospitals.'},
  electronics:  {name:'Electronics',     ico:'📻', era:6, cost:282, req:['electricity'], note:'+2 science per city.', flat:{sci:2}},
  flight:       {name:'Flight',          ico:'✈️', era:6, cost:300, req:['combustion'], note:'+2 gold per city.', flat:{gold:2}},
  // ---- Information Age (7) ----
  computing:    {name:'Computing',       ico:'💻', era:7, cost:360, req:['electronics'], unlock:['datacenter'], note:'Data Centers. +1 science/city.', flat:{sci:1}},
  internet:     {name:'The Internet',    ico:'🌐', era:7, cost:420, req:['computing'], unlock:['fiber'], note:'Fiber & 5G. +1 gold/city.', flat:{gold:1}},
  renewables:   {name:'Renewable Energy',ico:'🔆', era:7, cost:420, req:['computing','electricity'], unlock:['solar'], note:'Renewable Grid. Cuts climate risk.'},
  automation:   {name:'Automation',      ico:'🦾', era:7, cost:460, req:['computing','combustion'], note:'+2 production per city.', flat:{prod:2}},
  fintech:      {name:'FinTech',         ico:'💳', era:7, cost:460, req:['internet','banking'], unlock:['exchange'], note:'Stock Exchanges. Stability.'},
  genomics:     {name:'Genomics',        ico:'🧫', era:7, cost:440, req:['medicine','computing'], note:'+health. Basis of biotech.', nat:{health:10}},
  // ---- AI Age (8) ----
  machine_learning:{name:'Machine Learning', ico:'🤖', era:8, cost:560, req:['internet','automation'], unlock:['ailab'], note:'AI Labs. +2 science/city.', flat:{sci:2}},
  biotech:      {name:'Biotechnology',   ico:'🧬', era:8, cost:620, req:['genomics','internet'], unlock:['biolab'], note:'Biotech Labs. Pandemic defence.'},
  cybersecurity:{name:'Cybersecurity',   ico:'🛡️', era:8, cost:620, req:['fintech'], unlock:['cyber'], note:'Cyber Command.'},
  big_data:     {name:'Big Data',        ico:'📊', era:8, cost:680, req:['machine_learning'], note:'+2 gold & +2 science per city.', flat:{gold:2, sci:2}},
  quantum:      {name:'Quantum Computing',ico:'⚛️', era:8, cost:760, req:['machine_learning'], unlock:['quantum'], note:'Quantum Hubs. Massive science.'},
  // ---- Singularity Age (9) ----
  fusion:       {name:'Fusion Power',    ico:'☢️', era:9, cost:900,  req:['quantum','renewables'], unlock:['fusion'], note:'Fusion Plants. Clean energy.'},
  nanotech:     {name:'Nanotechnology',  ico:'🔬', era:9, cost:950,  req:['biotech','quantum'], note:'+3 production & +health everywhere.', flat:{prod:3}, nat:{health:10}},
  agi:          {name:'Artificial General Intelligence', ico:'🧠', era:9, cost:1100, req:['quantum','big_data'], note:'+4 science per city. The brink.', flat:{sci:4}},
  singularity:  {name:'The Singularity', ico:'✨', era:9, cost:1500, req:['agi','fusion','nanotech'], note:'WIN THE GAME. Transcend.', win:true},
};

/* ----------------------------- State ----------------------------- */
let S = null;
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

function newGame(civ){
  civ = civ || DEFAULT_CIV;
  const map = genMap();
  S = {
    turn: 1, era: 0,
    civ: {name:civ.name, ico:civ.ico, color:civ.color, b:civ.b||{}},
    treasury: 50 + (civ.b && civ.b.startGold || 0),
    sciStore: 0,
    health: 100, stability: 100,
    research: null,
    techs: {},                 // id -> true
    map,
    cities: [],
    log: [],
    bankruptTurns: 0,
    over: false, won: false,
    cache: {},
  };
  // Found the capital on a good land tile near the centre.
  const start = bestStartTile(map);
  foundCity(start, 'Capital', true);
  // Begin researching the foundational tech so science is never wasted;
  // the player can switch targets any time from the Tech tab.
  S.research = 'agriculture';
  pushLog(`The ${S.civ.name} is founded in the Stone Age. Lead it across the ages.`, 'good');
  pushLog('Research has begun on Agriculture. Tap the Tech tab to change it.', 'tech');
  recompute();
}

/* ----------------------------- Civ bonus helpers ----------------------------- */
function civB(k, def){ return (S && S.civ && S.civ.b && S.civ.b[k]!=null) ? S.civ.b[k] : def; }
function buildingCost(id){ return Math.max(1, Math.round(BUILDINGS[id].cost * civB('buildCostMul',1))); }
function settlerCost(){ return Math.max(8, Math.round(30 * civB('settleMul',1))); }
function foundGoldCost(){ return Math.round((40 + S.cities.length*15) * civB('settleMul',1)); }

/* ----------------------------- Map gen ----------------------------- */
function idx(x,y){ return y*MAP_W + x; }
function inBounds(x,y){ return x>=0 && x<MAP_W && y>=0 && y<MAP_H; }

function genMap(){
  const map = [];
  for(let y=0;y<MAP_H;y++){
    for(let x=0;x<MAP_W;x++){
      const edge = (x===0||y===0||x===MAP_W-1||y===MAP_H-1);
      let terrain;
      const r = Math.random();
      if(edge && r<0.55) terrain = r<0.3?'ocean':'coast';
      else if(r<0.12) terrain = 'coast';
      else {
        const w = Math.random();
        terrain = w<0.30?'grass': w<0.52?'plains': w<0.68?'forest':
                  w<0.80?'hills': w<0.88?'mountain': w<0.95?'desert':'tundra';
      }
      map.push({x,y,terrain,owner:false,cityId:null});
    }
  }
  return map;
}

function bestStartTile(map){
  let best=null, bestScore=-1;
  for(const t of map){
    if(!TERRAIN[t.terrain].land) continue;
    const dx = Math.abs(t.x-(MAP_W-1)/2), dy = Math.abs(t.y-(MAP_H-1)/2);
    let score = TERRAIN[t.terrain].food*2 + TERRAIN[t.terrain].prod;
    for(const n of neighbors(map,t.x,t.y)) score += TERRAIN[n.terrain].food + TERRAIN[n.terrain].prod*0.5;
    score -= (dx+dy)*0.6;
    if(score>bestScore){bestScore=score;best=t;}
  }
  return best;
}

function neighbors(map,x,y){
  const out=[];
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(dx===0&&dy===0)continue;
    if(inBounds(x+dx,y+dy)) out.push(map[idx(x+dx,y+dy)]);
  }
  return out;
}
function ring(map,x,y){ return [map[idx(x,y)], ...neighbors(map,x,y)]; }

/* ----------------------------- Cities ----------------------------- */
let cityCounter = 0;
const CITY_NAMES = ['Capital','Neo Harbor','Silicon Bay','Aurora','Quantum City',
  'Helios','Nova Reach','Synth Falls','Vertex','Datia','Lumen','Orbital'];

function foundCity(tile, name, isCapital){
  const id = ++cityCounter;
  const city = {
    id, x:tile.x, y:tile.y,
    name: name || CITY_NAMES[Math.min(S.cities.length, CITY_NAMES.length-1)],
    pop: isCapital?3:1,
    foodStore:0, prodStore:0,
    buildings: {}, building: null,
    capital: !!isCapital,
  };
  tile.cityId = id;
  for(const t of ring(S.map, tile.x, tile.y)) t.owner = true;
  S.cities.push(city);
  return city;
}
function cityById(id){ return S.cities.find(c=>c.id===id); }

function cityWorkedTiles(c){
  const r = ring(S.map, c.x, c.y);
  const center = S.map[idx(c.x,c.y)];
  const others = r.filter(t=>t!==center && t.owner)
    .sort((a,b)=> tileValue(b)-tileValue(a))
    .slice(0, c.pop);
  return [center, ...others];
}
function tileValue(t){ const T=TERRAIN[t.terrain]; return T.food*2+T.prod*1.5+T.gold+T.sci; }

// Per-city raw yields (after civ multipliers, before national meter penalties).
function cityYields(c){
  let food=0,prod=0,gold=0,sci=0;
  for(const t of cityWorkedTiles(c)){
    const T=TERRAIN[t.terrain]; food+=T.food; prod+=T.prod; gold+=T.gold; sci+=T.sci;
  }
  for(const b in c.buildings){
    const eff = BUILDINGS[b].city||{};
    food+=eff.food||0; prod+=eff.prod||0; gold+=eff.gold||0; sci+=eff.sci||0;
  }
  for(const tid in S.techs){
    const f = TECHS[tid].flat; if(f){food+=f.food||0;prod+=f.prod||0;gold+=f.gold||0;sci+=f.sci||0;}
  }
  gold += Math.floor(c.pop/2);
  sci  += Math.ceil(c.pop/2);
  // civilization multipliers
  food = Math.round(food*civB('foodMul',1));
  prod = Math.round(prod*civB('prodMul',1));
  gold = Math.round(gold*civB('goldMul',1));
  sci  = Math.round(sci *civB('sciMul',1));
  food -= c.pop*2;           // citizens eat
  return {food,prod,gold,sci};
}

/* ----------------------------- National compute ----------------------------- */
function nationalFlags(){
  const f = {health:0,stab:0,pandemicResist:0,crisisResist:0,climateResist:0,cyberResist:0};
  for(const c of S.cities) for(const b in c.buildings){
    const nat = BUILDINGS[b].nat; if(nat) for(const k in nat) f[k]+=nat[k];
  }
  for(const tid in S.techs){ const nat=TECHS[tid].nat; if(nat) for(const k in nat) f[k]+=nat[k]; }
  return f;
}
function maintenance(){
  let m=0; for(const c of S.cities) for(const b in c.buildings) m+=BUILDINGS[b].maint;
  return m;
}
function recompute(){
  let goldIncome=0, sciTotal=0, pop=0;
  for(const c of S.cities){
    const y = cityYields(c);
    goldIncome += y.gold; sciTotal += y.sci; pop += c.pop;
  }
  goldIncome -= maintenance();
  if(S.stability<40) goldIncome = Math.round(goldIncome*0.7);
  if(S.health<40)    sciTotal   = Math.round(sciTotal*0.8);
  S.cache = {goldIncome, sciTotal, pop, flags:nationalFlags(), maint:maintenance()};
}
function eraName(){ return ERAS[S.era]; }
function highestEraResearched(){
  let e=0; for(const t in S.techs) e=Math.max(e,TECHS[t].era); return e;
}

/* ----------------------------- Turn loop ----------------------------- */
function endTurn(){
  if(S.over) return;
  for(const c of S.cities){
    const y = cityYields(c);
    c.foodStore += y.food;
    const growCost = GROW_BASE * c.pop;
    if(c.foodStore >= growCost){ c.pop++; c.foodStore -= growCost; pushLog(`${c.name} grew to population ${c.pop}.`); }
    else if(c.foodStore < 0){
      if(c.pop>1){ c.pop--; c.foodStore=0; pushLog(`${c.name} suffered famine — population fell to ${c.pop}.`,'bad'); }
      else c.foodStore=0;
    }
    if(c.building){
      c.prodStore += Math.max(0,y.prod);
      const target = c.building.kind==='building' ? buildingCost(c.building.id) : c.building.cost;
      if(c.prodStore >= target) completeBuild(c);
    }
  }
  recompute();
  S.treasury += S.cache.goldIncome;
  S.sciStore += S.cache.sciTotal;

  if(S.research){
    const t = TECHS[S.research];
    if(S.sciStore >= t.cost){ S.sciStore -= t.cost; completeTech(S.research); }
  }

  driftMeters();

  if(S.treasury < 0){
    S.bankruptTurns++;
    S.stability = Math.max(0, S.stability-8);
    pushLog('Treasury is in deficit! Stability falling.', 'bad');
    if(S.bankruptTurns>=4) return gameOver(false,'Your economy collapsed under unsustainable debt.');
  } else S.bankruptTurns=0;

  rollEvent();

  const e = highestEraResearched();
  if(e>S.era){ S.era=e; pushLog(`Your civilization has entered the ${ERAS[e]}!`, 'tech'); toast(`🚀 Welcome to the ${ERAS[e]}`); }

  if(S.cities.length===0) return gameOver(false,'Your last city is gone. The civilization is no more.');
  S.turn++;
  if(S.turn>WIN_TURN && !S.over) return scoreVictory();

  recompute();
  render();
}

function driftMeters(){
  const f = S.cache.flags;
  const popPressure = Math.min(40, S.cache.pop*0.8);
  const healthEq = clamp(72 - popPressure + f.health + civB('healthBonus',0), 10, 100);
  const stabEq   = clamp(75 - S.cities.length*2 + f.stab + civB('stabBonus',0) + (S.treasury>0?6:-10), 10, 100);
  S.health    += Math.sign(healthEq-S.health) * Math.min(4, Math.abs(healthEq-S.health));
  S.stability += Math.sign(stabEq-S.stability) * Math.min(4, Math.abs(stabEq-S.stability));
  S.health=clamp(S.health,0,100); S.stability=clamp(S.stability,0,100);
}

/* ----------------------------- Build system ----------------------------- */
function setBuild(c, kind, id){
  if(kind==='building'){ c.building={kind,id}; c.prodStore=Math.min(c.prodStore, buildingCost(id)); }
  else if(kind==='settler'){ c.building={kind:'settler', id:'settler', cost:settlerCost()}; c.prodStore=Math.min(c.prodStore, settlerCost()); }
  render();
}
function completeBuild(c){
  const b=c.building;
  if(b.kind==='building'){
    c.buildings[b.id]=true;
    pushLog(`${c.name} completed ${BUILDINGS[b.id].name}.`, 'good');
    c.prodStore=0; c.building=null;
  } else if(b.kind==='settler'){
    const spot = findExpansionTile();
    c.prodStore=0; c.building=null;
    if(spot){ const nc=foundCity(spot); pushLog(`Settlers founded ${nc.name}!`, 'good'); toast(`🏙️ Founded ${nc.name}`); }
    else pushLog(`${c.name} trained settlers but found no land to settle.`, 'warn');
  }
}
function findExpansionTile(){
  const cands=[];
  for(const t of S.map){
    if(!TERRAIN[t.terrain].land || t.cityId) continue;
    if(S.cities.some(c=>Math.abs(c.x-t.x)<=1 && Math.abs(c.y-t.y)<=1)) continue;
    const touchesEmpire = ring(S.map,t.x,t.y).some(n=>n.owner) ||
                          S.cities.some(c=>Math.abs(c.x-t.x)<=3 && Math.abs(c.y-t.y)<=3);
    if(touchesEmpire) cands.push(t);
  }
  if(!cands.length) return null;
  cands.sort((a,b)=>tileValue(b)-tileValue(a));
  return cands[0];
}

/* ----------------------------- Research ----------------------------- */
function canResearch(id){
  if(S.techs[id]) return false;
  return TECHS[id].req.every(r=>S.techs[r]);
}
function chooseResearch(id){
  if(!canResearch(id)) return;
  S.research=id; closeModal(); render();
  toast(`🔬 Researching ${TECHS[id].name}`);
}
function completeTech(id){
  S.techs[id]=true;
  const t=TECHS[id];
  pushLog(`Researched ${t.name}! ${t.note}`, 'tech');
  toast(`✅ ${t.name} discovered`);
  if(t.win) return gameOver(true,'You achieved the Technological Singularity. Your civilization transcends!');
  S.research=null;
  const next = Object.keys(TECHS).filter(canResearch).sort((a,b)=>TECHS[a].cost-TECHS[b].cost)[0];
  if(next) S.research=next;
}

/* ----------------------------- Events ----------------------------- */
function rollEvent(){
  const f=S.cache.flags;
  const eraMul = 1 + S.era*0.08;            // modern world = more systemic risk
  const events = [];
  const pandemicP = clamp((0.08 + (60-S.health)/240) * eraMul - f.pandemicResist*0.05, 0, 0.55);
  const crisisP   = clamp((0.07 + (60-S.stability)/240) * eraMul - f.crisisResist*0.05, 0, 0.55);
  const cyberP    = clamp((S.era>=7?0.10:0) - f.cyberResist*0.05, 0, 0.4);   // only once computers exist
  const climateP  = clamp((S.era>=5?0.07:0.03)*eraMul - f.climateResist*0.04, 0, 0.4);

  if(Math.random()<pandemicP) events.push('pandemic');
  if(Math.random()<crisisP)   events.push('crisis');
  if(Math.random()<cyberP)    events.push('cyber');
  if(Math.random()<climateP)  events.push('climate');
  if(Math.random()<0.13)      events.push(pick(['boom','aiwin','goldenage','migration']));

  for(const e of events.slice(0,2)) EVENTS[e]();
}

const EVENTS = {
  pandemic(){
    const sev = 1 + (S.health<40?1:0);
    const lost = Math.max(1, Math.round(S.cache.pop*0.10*sev));
    spreadPopLoss(lost);
    S.health = clamp(S.health-18*sev,0,100);
    S.treasury -= 10*sev;
    const what = S.era<6 ? 'A PLAGUE' : 'A GLOBAL PANDEMIC';
    pushLog(`🦠 ${what}: ${lost} population lost, health and economy hit.`, 'bad');
    toast('🦠 A pandemic sweeps the land');
    decision('🦠 Pandemic',
      'A deadly disease is spreading. How do you respond?',
      [
        {t:'Fund a cure (−20 gold)', d:'+10 health, +30 science', f:()=>{S.treasury-=20;S.health=clamp(S.health+10,0,100);S.sciStore+=30;}},
        {t:'Quarantine the cities', d:'+14 health, −12 stability', f:()=>{S.health=clamp(S.health+14,0,100);S.stability=clamp(S.stability-12,0,100);}},
        {t:'Endure it', d:'No action', f:()=>{}},
      ]);
  },
  crisis(){
    const loss = Math.round(Math.max(15, S.treasury*0.25));
    S.treasury -= loss;
    S.stability = clamp(S.stability-20,0,100);
    const what = S.era<5 ? 'A TRADE COLLAPSE' : 'A FINANCIAL CRISIS';
    pushLog(`📉 ${what}: markets crash, −${loss} gold, stability plunges.`, 'bad');
    toast('📉 Financial crisis!');
    decision('📉 Economic Crisis',
      'The economy is in free-fall. What is your policy?',
      [
        {t:'Bail out the banks (−25 gold)', d:'+18 stability', f:()=>{S.treasury-=25;S.stability=clamp(S.stability+18,0,100);}},
        {t:'Stimulus spending (−15 gold)', d:'+10 stability, +1 pop to capital', f:()=>{S.treasury-=15;S.stability=clamp(S.stability+10,0,100);if(S.cities[0])S.cities[0].pop++;}},
        {t:'Austerity', d:'+8 stability, −20 science', f:()=>{S.stability=clamp(S.stability+8,0,100);S.sciStore=Math.max(0,S.sciStore-20);}},
      ]);
  },
  cyber(){
    const stolen = Math.round(Math.max(10, S.treasury*0.15));
    S.treasury -= stolen; S.sciStore=Math.max(0,S.sciStore-15);
    pushLog(`💻 CYBER-ATTACK: hackers stole ${stolen} gold and disrupted research.`, 'bad');
    toast('💻 Cyber-attack!');
  },
  climate(){
    spreadPopLoss(Math.max(1,Math.round(S.cache.pop*0.05)));
    S.treasury -= 12; S.stability=clamp(S.stability-8,0,100);
    const what = S.era<5 ? 'A GREAT FLOOD' : 'A CLIMATE DISASTER';
    pushLog(`🌪️ ${what}: extreme weather damages cities and crops.`, 'bad');
    toast('🌪️ Natural disaster');
  },
  boom(){ const g=12+S.era*6; S.treasury+=g; pushLog(`📈 ECONOMIC BOOM: +${g} gold from a trade surge.`, 'good'); },
  aiwin(){ const s=15+S.era*12; S.sciStore+=s; const w=S.era>=8?'AI BREAKTHROUGH':'A GREAT DISCOVERY'; pushLog(`🔬 ${w}: +${s} science.`, 'good'); },
  goldenage(){ S.health=clamp(S.health+10,0,100);S.stability=clamp(S.stability+10,0,100); pushLog('✨ GOLDEN AGE: optimism lifts health & stability.', 'good'); },
  migration(){ if(S.cities.length){pick(S.cities).pop++; pushLog('👥 MIGRATION: newcomers boost a city.', 'good');} },
};

function spreadPopLoss(n){
  let left=n, guard=0;
  while(left>0 && S.cities.length && guard++<200){
    const c=pick(S.cities);
    if(c.pop>1){ c.pop--; left--; }
    else if(S.cities.length>1){
      const t=S.map[idx(c.x,c.y)]; t.cityId=null;
      S.cities=S.cities.filter(x=>x!==c); left--;
      pushLog('A city was abandoned.', 'bad');
    } else { left--; }   // never wipe the very last city outright
  }
}

/* ----------------------------- End states ----------------------------- */
function gameOver(won, msg){
  S.over=true; S.won=won; S.endMsg=msg; S.score=computeScore();
  render(); switchView('map');
}
function scoreVictory(){ gameOver(true, `You reached turn ${WIN_TURN} and led a thriving civilization into the future!`); }
function computeScore(){
  let s=0;
  s += S.cache.pop*10;
  s += Object.keys(S.techs).length*40;
  s += S.cities.length*25;
  s += Math.max(0,S.treasury);
  s += (S.health+S.stability);
  s += S.won?500:0;
  return Math.round(s);
}

/* ----------------------------- Helpers ----------------------------- */
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pick(arr){return arr[Math.floor(Math.random()*arr.length)];}
function pushLog(text,kind){ S.log.unshift({turn:S.turn,text,kind:kind||''}); if(S.log.length>120)S.log.pop(); }

let toastTimer=null;
function toast(msg){
  const el=$('#toast'); if(!el) return;
  el.textContent=msg; el.classList.remove('hidden');
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.classList.add('hidden'),2400);
}

/* ----------------------------- Modal ----------------------------- */
function openModal(title, bodyHTML, onMount){
  $('#modal-title').textContent=title;
  $('#modal-body').innerHTML=bodyHTML;
  $('#modal-backdrop').classList.remove('hidden');
  if(onMount) onMount($('#modal-body'));
}
function closeModal(){ $('#modal-backdrop').classList.add('hidden'); }

let decisionQueue=[];
function decision(title, text, options){ decisionQueue.push({title,text,options}); }
function flushDecisions(){
  if(!decisionQueue.length || S.over) return;
  const d=decisionQueue.shift();
  const html = `<p class="muted" style="margin-top:0">${d.text}</p>` +
    d.options.map((o,i)=>`<div class="opt">
        <div><div class="ot">${o.t}</div><div class="odesc">${o.d||''}</div></div>
        <button class="btn go" data-i="${i}">Choose</button></div>`).join('');
  openModal(d.title, html, (root)=>{
    root.querySelectorAll('button[data-i]').forEach(el=>el.onclick=()=>{
      const opt=d.options[+el.dataset.i]; opt.f&&opt.f();
      recompute(); closeModal(); render();
      setTimeout(flushDecisions, 120);
    });
  });
}

/* ===================================================================
   SETUP / CIVILIZATION SELECTION
   =================================================================== */
let setupSel=null, customFocus=null;
function bonusText(b){
  const p=[];
  if(b.foodMul) p.push(`+${Math.round((b.foodMul-1)*100)}% food`);
  if(b.prodMul) p.push(`+${Math.round((b.prodMul-1)*100)}% production`);
  if(b.goldMul) p.push(`+${Math.round((b.goldMul-1)*100)}% gold`);
  if(b.sciMul)  p.push(`+${Math.round((b.sciMul-1)*100)}% science`);
  if(b.buildCostMul) p.push(`${Math.round((1-b.buildCostMul)*100)}% cheaper builds`);
  if(b.settleMul) p.push(`${Math.round((1-b.settleMul)*100)}% cheaper settling`);
  if(b.stabBonus) p.push(`+${b.stabBonus} stability`);
  if(b.healthBonus) p.push(`+${b.healthBonus} health`);
  if(b.startGold) p.push(`+${b.startGold} start gold`);
  return p.join(' · ') || 'Balanced';
}

function showSetup(){
  const el=$('#setup'); if(!el) return;
  setupSel=null; customFocus=null;
  const cards = Object.entries(CIVS).map(([k,c])=>`
    <div class="civ-card" data-civ="${k}" style="--cc:${c.color}">
      <div class="civ-ico">${c.ico}</div>
      <div class="civ-name">${c.name}</div>
      <div class="civ-blurb">${c.blurb}</div>
      <div class="civ-bonus">${bonusText(c.b)}</div>
    </div>`).join('');
  const chips = Object.entries(FOCUSES).map(([k,f])=>
    `<button class="chip" data-focus="${k}">${f.label}</button>`).join('');
  el.innerHTML = `
    <div class="setup-inner">
      <div class="setup-hero">
        <div class="setup-logo">🌍 → ✨</div>
        <h1>Civilization 2026</h1>
        <p>Lead a people from the <b>Stone Age</b> to the <b>Singularity</b> — through writing,
           currency, gunpowder, electricity, the Internet, AI and fusion.</p>
        <p class="muted">Choose your civilization to begin.</p>
      </div>
      <div class="civ-grid">
        ${cards}
        <div class="civ-card custom" data-civ="__custom" style="--cc:#7ee0ff">
          <div class="civ-ico">✨</div>
          <div class="civ-name">Create Your Own</div>
          <div class="civ-blurb">Name your civilization and choose its strength.</div>
        </div>
      </div>
      <div id="custom-form" class="hidden">
        <input id="custom-name" type="text" placeholder="Name your civilization" maxlength="24" />
        <div class="muted" style="margin:8px 0 4px">Choose a strength:</div>
        <div class="focus-chips">${chips}</div>
      </div>
      <button id="start-game" class="btn go full" disabled>Choose a civilization</button>
    </div>`;

  const startBtn=$('#start-game');
  function refresh(){
    $$('#setup .civ-card').forEach(c=>c.classList.toggle('sel', c.dataset.civ===setupSel));
    $$('#setup .chip').forEach(c=>c.classList.toggle('sel', c.dataset.focus===customFocus));
    const isCustom = setupSel==='__custom';
    $('#custom-form').classList.toggle('hidden', !isCustom);
    if(!setupSel){ startBtn.disabled=true; startBtn.textContent='Choose a civilization'; }
    else if(isCustom && !customFocus){ startBtn.disabled=true; startBtn.textContent='Pick a strength'; }
    else { startBtn.disabled=false;
      const nm = isCustom ? (($('#custom-name').value||'').trim()||'New Republic') : CIVS[setupSel].name;
      startBtn.textContent='Begin as the '+nm+' ▶'; }
  }
  $$('#setup .civ-card').forEach(c=>c.onclick=()=>{ setupSel=c.dataset.civ; refresh(); });
  $$('#setup .chip').forEach(c=>c.onclick=()=>{ customFocus=c.dataset.focus; refresh(); });
  const nameInput=$('#custom-name'); if(nameInput) nameInput.oninput=refresh;
  startBtn.onclick=()=>{
    let civ;
    if(setupSel==='__custom'){
      const nm=(($('#custom-name').value||'').trim())||'New Republic';
      civ={name:nm, ico:FOCUSES[customFocus].label.split(' ')[0], color:'#7ee0ff', b:FOCUSES[customFocus].b};
    } else civ=CIVS[setupSel];
    startGame(civ);
  };
  el.classList.remove('hidden');
  refresh();
}
function startGame(civ){
  $('#setup').classList.add('hidden');
  newGame(civ);
  switchView('map');
  render();
}

/* ===================================================================
   RENDERING
   =================================================================== */
function render(){
  if(!S) return;
  renderTop(); renderResearchBanner();
  renderMap(); renderCities(); renderTech(); renderLog();
  setTimeout(flushDecisions, 60);
}

function renderTop(){
  recompute();
  $('#val-era').textContent=(S.civ?S.civ.ico+' ':'')+eraName();
  $('#val-turn').textContent=`${S.turn}/${WIN_TURN}`;
  $('#val-gold').textContent=S.treasury;
  const d=$('#val-gold-delta'); const gi=S.cache.goldIncome;
  d.textContent=` (${gi>=0?'+':''}${gi})`; d.className='delta '+(gi>=0?'pos':'neg');
  $('#val-sci').textContent='+'+S.cache.sciTotal;
  $('#val-pop').textContent=S.cache.pop;
  $('#val-health').textContent=Math.round(S.health);
  $('#val-stab').textContent=Math.round(S.stability);
}

function renderResearchBanner(){
  if(S.research){
    const t=TECHS[S.research];
    $('#rb-name').textContent=`${t.ico} ${t.name}`;
    const pct=clamp(S.sciStore/t.cost*100,0,100);
    $('#rb-fill').style.width=pct+'%';
    const remain=Math.max(0,t.cost-S.sciStore);
    const eta=S.cache.sciTotal>0?Math.ceil(remain/S.cache.sciTotal):'∞';
    $('#rb-eta').textContent=`${remain} sci · ${eta} turns`;
  } else {
    $('#rb-name').textContent='— tap to choose research —';
    $('#rb-fill').style.width='0%'; $('#rb-eta').textContent='';
  }
}

/* ---- Map view ---- */
function renderMap(){
  const v=$('#view-map');
  if(S.over) return renderEndScreen(v);
  const ts = Math.max(34, Math.min(54, Math.floor((Math.min(window.innerWidth,520)-40)/MAP_W)));
  let html=`<div class="map-hint">Tap an empire tile near your border to expand, or a city to manage it.</div>
    <div id="map-wrap"><div id="map-grid" style="grid-template-columns:repeat(${MAP_W},var(--ts));--ts:${ts}px">`;
  for(const t of S.map){
    const T=TERRAIN[t.terrain];
    const cls=['tile',T.cls];
    if(t.owner)cls.push('owned');
    if(t.cityId)cls.push('city');
    const city = t.cityId?cityById(t.cityId):null;
    html+=`<div class="${cls.join(' ')}" data-x="${t.x}" data-y="${t.y}">
      ${city?`🏙️<span class="citydot">${city.pop}</span>`:T.ico}
    </div>`;
  }
  html+=`</div></div>
    <div class="map-legend">
      <span>🏙️ City</span><span style="color:var(--accent)">▣ Your land</span>
      <span>${TERRAIN.grass.ico} Grass</span><span>${TERRAIN.forest.ico} Forest</span>
      <span>${TERRAIN.hills.ico} Hills</span><span>${TERRAIN.mountain.ico} Mtn</span>
      <span>${TERRAIN.coast.ico} Coast</span>
    </div>`;
  v.innerHTML=html;
  v.querySelectorAll('.tile').forEach(el=>el.onclick=()=>onTileTap(+el.dataset.x,+el.dataset.y));
}

function onTileTap(x,y){
  const t=S.map[idx(x,y)];
  if(t.cityId) return openCity(cityById(t.cityId));
  if(t.owner && TERRAIN[t.terrain].land){
    const tooClose = S.cities.some(c=>Math.abs(c.x-x)<=1 && Math.abs(c.y-y)<=1);
    const cost = foundGoldCost();
    openModal(`${TERRAIN[t.terrain].ico} ${TERRAIN[t.terrain].name}`,
      `<p class="muted">Yields: 🌾${TERRAIN[t.terrain].food} 🏭${TERRAIN[t.terrain].prod} 💰${TERRAIN[t.terrain].gold} 🔬${TERRAIN[t.terrain].sci}</p>
       ${tooClose?'<p class="muted">Too close to an existing city to settle here.</p>':
         `<button class="btn gold full" id="settle">🏙️ Found city here (−${cost} gold)</button>`}`,
      (root)=>{ const b=root.querySelector('#settle'); if(b) b.onclick=()=>{
        if(S.treasury<cost){ toast('Not enough gold'); return; }
        S.treasury-=cost; const nc=foundCity(t); closeModal();
        pushLog(`Founded ${nc.name} for ${cost} gold.`, 'good'); toast(`🏙️ Founded ${nc.name}`); render();
      };});
    return;
  }
  openModal(`${TERRAIN[t.terrain].ico} ${TERRAIN[t.terrain].name}`,
    `<p class="muted">Outside your borders. Build Settlers in a city, or expand from a bordering empire tile.</p>
     <p class="muted">Yields: 🌾${TERRAIN[t.terrain].food} 🏭${TERRAIN[t.terrain].prod} 💰${TERRAIN[t.terrain].gold} 🔬${TERRAIN[t.terrain].sci}</p>`);
}

/* ---- Cities view ---- */
function renderCities(){
  const v=$('#view-cities');
  if(S.over) return;
  if(!S.cities.length){ v.innerHTML='<div class="card">No cities.</div>'; return; }
  v.innerHTML = S.cities.map(c=>{
    const y=cityYields(c);
    const buildLabel = c.building ?
      (c.building.kind==='building'?BUILDINGS[c.building.id].name:'Settlers') : '— idle —';
    const target = c.building?(c.building.kind==='building'?buildingCost(c.building.id):c.building.cost):0;
    const pct = target?clamp(c.prodStore/target*100,0,100):0;
    const growCost=GROW_BASE*c.pop;
    return `<div class="card">
      <h3>🏙️ ${c.name} ${c.capital?'<span class="pill">★ Capital</span>':''}
        <span style="margin-left:auto" class="pill">Pop ${c.pop}</span></h3>
      <div class="yields">
        <span class="y-food">🌾 ${y.food>=0?'+':''}${y.food}</span>
        <span class="y-prod">🏭 ${y.prod}</span>
        <span class="y-gold">💰 ${y.gold}</span>
        <span class="y-sci">🔬 ${y.sci}</span>
      </div>
      <div class="row"><span class="muted">Growth</span><span class="muted">${c.foodStore}/${growCost} food</span></div>
      <div class="row"><span class="muted">Building</span><span>${buildLabel}</span></div>
      ${c.building?`<div class="rb-bar" style="margin:4px 0"><div class="rb-fill" style="width:${pct}%"></div></div>
        <div class="muted">${c.prodStore}/${target} production</div>`:''}
      <button class="btn full go" data-city="${c.id}" style="margin-top:8px">Manage ${c.name}</button>
    </div>`;
  }).join('');
  v.querySelectorAll('[data-city]').forEach(el=>el.onclick=()=>openCity(cityById(+el.dataset.city)));
}

function openCity(c){
  const owned = Object.keys(c.buildings);
  const available = Object.keys(BUILDINGS).filter(b=>{
    if(c.buildings[b]) return false;
    const tech=BUILDINGS[b].tech; return !tech || S.techs[tech];
  });
  const futureLocked = Object.keys(BUILDINGS).filter(b=>{
    const tech=BUILDINGS[b].tech; return tech && !S.techs[tech] && !c.buildings[b];
  });
  const buildOpt = (b)=>{
    const B=BUILDINGS[b]; const active=c.building&&c.building.id===b;
    return `<div class="opt">
      <div><div class="ot">${B.ico} ${B.name}</div><div class="odesc">${B.desc} · upkeep ${B.maint}💰</div></div>
      <button class="btn ${active?'gold':'go'}" data-build="${b}">${active?'Building…':`Build · ${buildingCost(b)}🏭`}</button>
    </div>`;
  };
  const yy=cityYields(c);
  const html = `
    <div class="yields" style="margin-bottom:8px">
      <span class="y-food">🌾 ${yy.food}</span><span class="y-prod">🏭 ${yy.prod}</span>
      <span class="y-gold">💰 ${yy.gold}</span><span class="y-sci">🔬 ${yy.sci}</span>
    </div>
    <div class="opt">
      <div><div class="ot">👥 Train Settlers</div><div class="odesc">Found a new city automatically on free nearby land.</div></div>
      <button class="btn ${c.building&&c.building.kind==='settler'?'gold':'go'}" data-build="__settler">${c.building&&c.building.kind==='settler'?'Training…':`Build · ${settlerCost()}🏭`}</button>
    </div>
    <div class="era-title">Available</div>
    ${available.length?available.map(buildOpt).join(''):'<p class="muted">All current buildings constructed. Research more tech!</p>'}
    ${owned.length?`<div class="era-title">Built</div>${owned.map(b=>`<div class="opt"><div><div class="ot">${BUILDINGS[b].ico} ${BUILDINGS[b].name}</div><div class="odesc">${BUILDINGS[b].desc} · upkeep ${BUILDINGS[b].maint}💰</div></div><button class="btn" data-demolish="${b}">Demolish</button></div>`).join('')}`:''}
    ${futureLocked.length?`<div class="era-title">Needs research</div>${futureLocked.map(b=>`<div class="opt" style="opacity:.55"><div><div class="ot">${BUILDINGS[b].ico} ${BUILDINGS[b].name}</div><div class="odesc">Requires ${TECHS[BUILDINGS[b].tech].name}</div></div><span class="pill">🔒</span></div>`).join('')}`:''}
  `;
  openModal(`🏙️ ${c.name}`, html, (root)=>{
    root.querySelectorAll('[data-build]').forEach(el=>el.onclick=()=>{
      const id=el.dataset.build;
      if(id==='__settler') setBuild(c,'settler'); else setBuild(c,'building',id);
      closeModal(); render(); toast('Build order set');
    });
    root.querySelectorAll('[data-demolish]').forEach(el=>el.onclick=()=>{
      const id=el.dataset.demolish; delete c.buildings[id];
      pushLog(`${c.name} demolished ${BUILDINGS[id].name} to cut upkeep.`, 'warn');
      recompute(); closeModal(); render(); toast('Building demolished');
    });
  });
}

/* ---- Tech view ---- */
function renderTech(){
  const v=$('#view-tech');
  if(S.over) return;
  let html='';
  for(let era=0;era<ERAS.length;era++){
    const techs=Object.keys(TECHS).filter(t=>TECHS[t].era===era);
    if(!techs.length) continue;
    const done = techs.filter(t=>S.techs[t]).length;
    html+=`<div class="era-group"><div class="era-title">${era<=S.era?'':'🔒 '}${ERAS[era]} <span class="muted">${done}/${techs.length}</span></div>`;
    for(const id of techs){
      const t=TECHS[id]; let cls='tech'; let status='';
      if(S.techs[id]){cls+=' done'; status='<span class="pill">✓</span>';}
      else if(S.research===id){cls+=' active'; status=`<span class="cost">${S.sciStore}/${t.cost}🔬</span>`;}
      else if(canResearch(id)){status=`<span class="cost">${t.cost}🔬</span>`;}
      else {cls+=' locked'; status=`<span class="muted">needs ${t.req.map(r=>TECHS[r].name).join(', ')}</span>`;}
      html+=`<div class="${cls}" ${canResearch(id)?`data-tech="${id}"`:''}>
        <div class="th"><span class="tn">${t.ico} ${t.name}</span>${status}</div>
        <div class="td">${t.note}${t.win?' 🏆':''}</div>
      </div>`;
    }
    html+='</div>';
  }
  v.innerHTML=html;
  v.querySelectorAll('[data-tech]').forEach(el=>el.onclick=()=>chooseResearch(el.dataset.tech));
}

function openResearchPicker(){
  if(S.over) return;
  switchView('tech');
  toast('Pick a technology to research');
}

/* ---- Log view ---- */
function renderLog(){
  const v=$('#view-log');
  if(S.over) return;
  if(!S.log.length){ v.innerHTML='<div class="card muted">No events yet. End your turn to advance time.</div>'; return; }
  v.innerHTML=S.log.map(l=>`<div class="logitem ${l.kind}"><span class="lt">Turn ${l.turn}</span><br>${l.text}</div>`).join('');
}

/* ---- End screen ---- */
function renderEndScreen(){
  const html=`<div class="center-screen">
    <h1>${S.won?'🏆 Victory':'💀 Defeat'}</h1>
    <p>${S.civ.ico} The ${S.civ.name}</p>
    <p>${S.endMsg||''}</p>
    <div class="big-num">${S.score}</div>
    <p class="muted">Final score · Turn ${S.turn} · ${eraName()}<br>
      ${Object.keys(S.techs).length} techs · ${S.cities.length} cities · ${S.cache.pop} population</p>
    <button class="btn go full" id="restart">▶ New Civilization</button>
  </div>`;
  ['#view-map','#view-cities','#view-tech','#view-log'].forEach(s=>$(s).innerHTML=html);
  $$('#restart').forEach(b=>b.onclick=()=>{ showSetup(); });
}

/* ----------------------------- Navigation ----------------------------- */
function switchView(name){
  $$('.view').forEach(v=>v.classList.remove('active'));
  $('#view-'+name).classList.add('active');
  $$('.tab').forEach(t=>t.classList.toggle('active', t.dataset.view===name));
}

/* ----------------------------- Wiring ----------------------------- */
function wire(){
  $$('.tab').forEach(t=>t.onclick=()=>switchView(t.dataset.view));
  $('#end-turn').onclick=()=>{ if(S && !S.over) endTurn(); };
  $('#research-banner').onclick=()=>openResearchPicker();
  $('#modal-close').onclick=closeModal;
  $('#modal-backdrop').onclick=(e)=>{ if(e.target.id==='modal-backdrop') closeModal(); };
  window.addEventListener('resize', ()=>{ if(S && !S.over) renderMap(); });
}

/* ----------------------------- Boot ----------------------------- */
wire();
showSetup();

if('serviceWorker' in navigator){
  window.addEventListener('load',()=>navigator.serviceWorker.register('sw.js').catch(()=>{}));
}
