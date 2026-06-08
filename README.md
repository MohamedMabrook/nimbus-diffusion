# Nimbus Diffusion

A free OFX diffusion plugin for DaVinci Resolve, by Mohamed Mabrok.

Makes your footage look like it was shot through a Black Pro-Mist, Pro-Mist, Glimmerglass or CineBloom filter. Works directly on the Color page.

Based on the spektrafilm diffusion algorithm by Andrea Volpato. Released free under GPL v3.

## What's in it

Two separate diffusion stages you can use independently or stack:

- **Lens** - simulates a filter on the camera lens at capture
- **Print** - simulates diffusion at the print/output stage
- **Mix** - blend the whole effect up or down without touching anything else
- Anamorphic Stretch and Chromatic Bloom in the Advanced section if you want to get weird with it

## How to install

Download the installer from the [releases page](https://github.com/MohamedMabrook/nimbus-diffusion/releases), run it as administrator, restart Resolve.

If the sliders look off after installing, go to Preferences > System > Memory and GPU and clear the OFX cache, then restart again.

## Building from source

You need CMake, Visual Studio with C++ workload, and Git.

```
cd ofx
install.bat
```

Run install.bat as administrator.

## Credits

The diffusion algorithm and PSF math come from spektrafilm by Andrea Volpato (GPLv3). The OFX plugin, two-stage design, anamorphic stretch and chromatic bloom are original.

## License

GPL v3. See LICENSE file.
