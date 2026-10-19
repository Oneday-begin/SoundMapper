# Third-party notices

The MIT license covers original SoundMap application code, not third-party datasets, recordings, model assets, or libraries.

| Component | Attribution and scope |
| --- | --- |
| MERT | [m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M); project records weights under CC BY-NC 4.0. Weights are not bundled. |
| AST | [MIT AudioSet AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593); obtain separately and review the pinned model terms. |
| EnCodec | [Meta EnCodec](https://github.com/facebookresearch/encodec), MIT code. The reviewed checkpoint card lacked an explicit weight-license field; no commercial redistribution rights are asserted. |
| Demucs | [Demucs](https://github.com/facebookresearch/demucs) and the separately provisioned checkpoint; consult source/checkpoint terms. |
| AudioSet ontology | `instrument-ontology.json` is an attributed derivative under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), based on [revision d417d32](https://github.com/audioset/ontology/tree/d417d32bf59c711abb5910fd2f76a0eb44697991). |
| Plotly.js | Generated locally from the pinned MIT-licensed Plotly package by setup.ps1; the generated distribution is excluded from source control. |
| FMA | [Free Music Archive dataset](https://github.com/mdeff/fma); recording licenses vary by track. Audio and dataset metadata are not bundled. |
| CCMSim | [CCMSim](https://github.com/pxaris/CCMSim); annotation audit only. Annotation license CC BY 4.0, code MIT; source-audio rights are separate. |
| Ollama / Qwen | Optional locally installed runtime/model; their respective terms apply. No weights bundled. |

Other dependencies retain their own licenses. Consult their distributions and pinned upstream model cards before redistribution or commercial use. Current technical notes are in README.md and RESULTS.md.
