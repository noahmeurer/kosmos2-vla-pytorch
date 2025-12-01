## Files

**`kosmos2_dit_policy.py`**
- Defines `Kosmos2Backbone`, a wrapper around the `microsoft/kosmos-2-patch14-224` VLM to encode (multi-camera) visual observations.
- Defines `ContextEmbeddings` to embed low-dimensional robot/state inputs (e.g. qpos) into the same feature space.
- Defines `Kosmos2DITPolicy`, which combines Kosmos-2 features, context embeddings, and a DiT-style diffusion head into a Vision-Language-Action policy.

**`dit_noise_net.py`**
- Defines `_DiTNoiseNet`, a diffusion transformer that predicts noise over action sequences conditioned on observations and diffusion timesteps.
- Implements the supporting components for this head (positional encodings, timestep MLP, transformer blocks, AdaLN-Zero conditioning).

## Architecture

**VLM Backbone:** Microsoft Kosmos-2
**Action Head**: Diffusion transformer with AdaLN-Zero conditioning

## Design Rational

* **Proven Foundation:** Selected based on the team's prior success with Kosmos-2 in single-camera settings, leveraging its lightweight architecture and native visual grounding capabilities.
* **Multi-View Batching Scheme:** Extended the single-stream backbone to support multiple camera views via a **batched inference pipeline**. Inputs are flattened to `[B*N]` for the forward pass and reshaped to `[B, N, T, H]` to enable multi-camera perception.
* **Learned Text Pooling:** Implemented a **learnable weighted average pooling mechanism** to condense language-grounded text tokens for each camera view. This reduces the sequence length to minimize the $O(S^2)$ attention cost in the action head while preserving view-specific language conditioning.
* * **DiT-Policy Architecture:** Implemented the diffusion-style action head with **AdaLN-Zero conditioning** rather than a vanilla diffusion transformer recipe. This was found to improve training stability and eliminated the need for extensive hyperparameter tuning.
