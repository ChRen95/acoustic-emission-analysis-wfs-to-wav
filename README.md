# Converter for Acoustic Emission .wfs files to .wav Files

Convert acoustic emission `.wfs` files to `.wav` files.

A simple example:

```python
import acoustic_emission_analysis.core as aem

wfs_file = aem.WFS("your_wfs_file.wfs")
wfs_file.save_wav("your_wav_file.wav")
```

## License

Licensed under the terms of the [MIT License](LICENSE)
