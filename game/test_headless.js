/* Headless smoke test: stub a minimal DOM, load the game, and simulate
   many turns to surface runtime errors and sanity-check the loop. */
const fs = require('fs');

function fakeEl(){
  return {
    _children:[], style:{}, dataset:{}, classList:{add(){},remove(){},toggle(){},contains(){return false;}},
    textContent:'', innerHTML:'',
    querySelector(){return fakeEl();}, querySelectorAll(){return [];},
    addEventListener(){}, scrollIntoView(){}, onclick:null,
  };
}
global.document = { querySelector(){return fakeEl();}, querySelectorAll(){return [];}, addEventListener(){} };
global.window = { innerWidth:390, innerHeight:800, addEventListener(){} };
global.navigator = {};
const realSetTimeout = setTimeout;
global.setTimeout = ()=>0;          // swallow deferred UI refresh / decision flush
global.clearTimeout = ()=>{};

const gameCode = fs.readFileSync(__dirname+'/game.js','utf8');

const testCode = `
  // ---- run the test inside the same scope as the game ----
  let errors = 0, techsDone = 0;
  // boot now opens the setup screen instead of starting a game, so start one
  // explicitly (rotate through a preset civ + the default to exercise bonuses).
  const civKeys = Object.keys(CIVS);
  const testCiv = (typeof CIV_OVERRIDE!=='undefined' && CIV_OVERRIDE) ? CIV_OVERRIDE : CIVS[civKeys[0]];
  newGame(testCiv);
  // Simulate up to WIN_TURN+10 turns or until game over.
  for(let i=0;i<WIN_TURN+10 && !S.over;i++){
    // resolve queued decisions deterministically (pick option 0)
    while(decisionQueue.length){ const d=decisionQueue.shift(); if(d.options[0]&&d.options[0].f) d.options[0].f(); }
    // act like a player: keep researching, keep each city building something, expand
    if(!S.research){ const n=Object.keys(TECHS).filter(canResearch).sort((a,b)=>TECHS[a].cost-TECHS[b].cost)[0]; if(n) S.research=n; }
    for(const c of S.cities){
      if(!c.building){
        const avail=Object.keys(BUILDINGS).filter(b=>!c.buildings[b] && (!BUILDINGS[b].tech||S.techs[BUILDINGS[b].tech]));
        if(avail.length) setBuild(c,'building',avail[0]);
        else if(S.cities.length<6 && S.treasury>60) setBuild(c,'settler');
      }
    }
    try { endTurn(); }
    catch(e){ errors++; console.log('ERROR turn', S.turn, e.message, e.stack.split('\\n')[1]); break; }
  }
  while(decisionQueue.length){ const d=decisionQueue.shift(); if(d.options[0]&&d.options[0].f) d.options[0].f(); }
  techsDone = Object.keys(S.techs).length;
  console.log(JSON.stringify({
    errors, finalTurn:S.turn, over:S.over, won:S.won,
    cities:S.cities.length, pop:S.cache.pop, treasury:S.treasury,
    techsDone, era:S.era, health:Math.round(S.health), stability:Math.round(S.stability),
    score:S.over?S.score:computeScore(), endMsg:S.endMsg||null,
    logCount:S.log.length
  }, null, 2));
`;

try { eval(gameCode + testCode); }
catch(e){ console.error('FATAL', e); process.exit(1); }
