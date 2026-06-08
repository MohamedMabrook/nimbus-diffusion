# Nimbus Diffusion

**Free OFX optical diffusion plugin for DaVinci Resolve**
by Mohamed Mabrok — released under GPL v3

Simulates the look of physical diffusion filters (Black Pro-Mist, Pro-Mist,
Glimmerglass, CineBloom) directly on the DaVinci Resolve Color page.

---

## Features

- **Lens stage** — capture-stage diffusion (lens filter on camera)
- **Print stage** — output-stage diffusion (enlarger/print filter)
- **Mix** — global wet/dry blend
- **Anamorphic Stretch** — oval PSF for anamorphic lens looks *(original)*
- **Chromatic Bloom** — R/B channel aberration on the bloom *(original)*

## Installation (Windows)

1. Right-click `install.bat` → **Run as administrator**
2. Restart DaVinci Resolve
3. If controls look wrong: Preferences → System → Memory and GPU → Clear OFX cache

## Credits

PSF algorithm based on [spektrafilm](https://github.com/AcademySoftwareFoundation/spektrafilm)
by Andrea Volpato (GPLv3).

OFX plugin, two-stage architecture, Anamorphic Stretch, and Chromatic Bloom
by Mohamed Mabrok (GPLv3).

## License

GNU General Public License v3 — see [LICENSE](LICENSE)
