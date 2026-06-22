# Civilization 2026 🌐

A turn-based strategy game in the spirit of *Civilization II*, reimagined for
2026. Begin in the **Stone Age** and lead your people across **ten eras of
history** — discovering writing, currency, gunpowder, electricity, the
Internet, AI and fusion — all the way to the **Singularity**. Along the way you
must survive the modern world's defining risks: global pandemics, financial
crises, cyber-war and climate shocks.

## Choose (or create) a civilization

You start by picking who you are. Each preset civilization has a distinct
strength, and they really do play differently — the science-focused Hellenic
League rushes to the Singularity faster, while the Imperium out-builds everyone.

| Civilization | Strength |
|---|---|
| 🏺 Kingdom of the Nile | +25% food — fast-growing cities |
| 🏛️ Hellenic League | +20% science |
| 🦅 The Imperium | +18% production, 10% cheaper buildings |
| 🐉 Silk Road Dynasty | +22% gold, bonus starting treasury |
| ⚓ Norse Clans | cheap, rapid expansion |
| ⛵ Maritime Republic | +gold and a permanent stability bonus |
| ✨ **Create Your Own** | name it and choose a strength (Agrarian, Industrious, Mercantile, Scholarly, Expansionist or Resilient) |

## The journey through history

Research drives everything, advancing you through **10 eras** and a **43-tech
tree**:

> Stone Age → Bronze Age → Classical → Medieval → Renaissance → Industrial →
> Modern → Information → AI → **Singularity**

Early techs (Agriculture, Pottery, Writing) come quickly; later ones (Quantum
Computing, AGI, Fusion) are the work of an age. Each unlocks new buildings —
from Granaries and Libraries to AI Labs, Quantum Hubs and Fusion Plants — and
permanent civilization-wide bonuses.

It's a self-contained web app (no build step, no dependencies) designed
**mobile-first for iPhone**, and installable as a PWA for full-screen, offline play.

## Play on your iPhone

Because this is a web game, you don't need the App Store, Xcode, or a developer
account — just a URL.

1. **Host the `game/` folder** anywhere that serves static files. Easiest options:
   - **GitHub Pages**: push this branch, enable Pages, and open
     `https://<user>.github.io/<repo>/game/`.
   - **Any static host** (Netlify, Vercel, Cloudflare Pages, etc.): point it at `game/`.
2. On your iPhone, open that URL in **Safari**.
3. Tap the **Share** button → **Add to Home Screen**. It now launches full-screen
   like a native app and works offline (a service worker caches everything).

### Run it locally to try it now
```bash
cd game
python3 -m http.server 8000
# then open http://localhost:8000 on your computer,
# or http://<your-computer-ip>:8000 on a phone on the same Wi-Fi
```

## How to play

- **🗺️ Map** — your civilization's territory. Tap a city to manage it; tap an
  owned border tile to found a new city (costs gold).
- **🏙️ Cities** — each city works the best surrounding tiles for **food**
  (growth), **production** (building), **gold**, and **science**. Choose what
  each city builds. Train **Settlers** to auto-expand onto good land.
- **🧪 Tech** — research drives everything. Progress through the 43-tech tree
  spanning all ten eras, from Agriculture to **The Singularity**. Each tech
  unlocks new buildings and permanent bonuses.
- **📜 Log** — a running history of growth, discoveries and world events.
- **⏭️ End Turn** — advance time. Yields are collected, cities grow, research
  progresses, and the world rolls the dice on a global event.

### The two meters that keep you up at night
- **❤️ Health** — falls as cities urbanize. When it's low, **pandemics** become
  likely and devastating. Build **Hospitals** and research **Genomics/Biotech**.
- **🏦 Stability** — your economy's resilience. When it's low, **financial
  crises** strike. Build **Stock Exchanges** and research **FinTech**.

### Modern risks (and how to counter them)
| Event | Effect | Counter |
|---|---|---|
| 🦠 Global Pandemic | population & health loss | Hospitals, Biotech, Genomics |
| 📉 Financial Crisis | treasury crash, stability loss | Stock Exchange, FinTech |
| 💻 Cyber-attack | steals gold & science | Cyber Command (Cybersecurity) |
| 🌪️ Climate Disaster | population & economic damage | Renewable Grid, Fusion |

Many events also present a **decision** — fund a vaccine, bail out the banks,
impose austerity — with real trade-offs.

### Winning and losing
- **🏆 Victory**: research **The Singularity**, *or* lead the strongest
  civilization to turn 240.
- **💀 Defeat**: run a treasury deficit for 4 straight turns (economic collapse),
  or lose your last city.

> Tip: don't over-build. Every building has **upkeep**. If you go into the red,
> the income indicator turns negative — cut spending or **Demolish** a building
> to recover.

## Project layout
```
game/
├── index.html            # app shell
├── style.css             # mobile-first UI
├── game.js               # all game logic & rendering (vanilla JS)
├── manifest.webmanifest  # PWA metadata (installable)
├── sw.js                 # service worker (offline support)
├── icon.svg              # app icon
└── test_headless.js      # Node smoke test: simulates full games
```

## Tests
```bash
cd game
node test_headless.js   # simulates a full game, checks for runtime errors
```
