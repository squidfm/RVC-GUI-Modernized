## What's new in the GUI

Everything is in place — close and restart the GUI to load it.

* **“Max quality (slower)”** checkbox, on by default. It does four things:
  * Runs everything in full precision.
  * Merges the RMVPE and crepe pitch trackers. Where they agree, it averages them for steadier pitch. Where one slips an octave, it takes the one that fits the surrounding pitch.
  * Makes the index search look through 64 clusters instead of 1.
  * Gives each chunk 4 seconds of context on each side, and converts anything under 100 seconds in one piece.
* **“Protect consonants/breaths”** slider, default 0.33. This is the upstream RVC feature that stops s/t/k sounds and breaths from going metallic.
* **“Volume envelope mix”** slider, default 1.0, which means off. Lower values make the output follow the input’s loudness dynamics. This one is a matter of taste, so it’s not part of Max quality.
* **Clipping guard**. Loud outputs are turned down slightly instead of clipping when saved.
* **Faster batch runs**. The index file and the RMVPE model now load once instead of for every line.
