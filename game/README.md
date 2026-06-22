# Civilization 2026 🌐

A turn-based strategy game in the spirit of *Civilization II*, reimagined for
2026. Lead a civilization out of the **Information Age**, through the **AI Age**,
and into the **Singularity** — while surviving the modern world's defining
risks: global pandemics, financial crises, cyber-war and climate shocks.

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
- **🧪 Tech** — research drives everything. Progress through a 15-tech tree:
  *Computing → The Internet → Machine Learning → Quantum → AGI → Fusion →
  Nanotech → **The Singularity***. Each tech unlocks new buildings and
  permanent bonuses.
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
  civilization to turn 120.
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
