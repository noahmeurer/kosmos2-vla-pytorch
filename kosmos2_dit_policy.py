import torch
import copy
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModelForVision2Seq, Kosmos2Model
import diffusers.optimization
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from dit_noise_net import _DiTNoiseNet

class Kosmos2Backbone(nn.Module):
    def __init__(self, n_cams):
        super().__init__()
        self.model = Kosmos2Model.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.n_cams = n_cams

    def forward(self, images, text_command=None) -> dict[torch.Tensor]:
        BN = len(images)
        N = self.n_cams
        assert BN % N == 0
        B = BN // N
        
        # Tokenize text_command and process visual frames
        inputs = self.processor(
            text=text_command, 
            images=images,
            return_tensors="pt",
            add_eos_token=True,
        )  # [B*N, T] where T is total number of image_tok + text_tok

        # Move processed inputs to device
        inputs = inputs.to(self.model.device)

        # Call forward VLM -> returns embeddings of final hidden layer
        out = self.model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=False,
        )  # [B*N, T, H]
        H = out.last_hidden_state.size(-1)
        T = out.last_hidden_state.size(1)
        
        # Unflatten the B dim along the N dim to get last_hidden_states per n_cam view.
        last_hidden_state = out.last_hidden_state.view(B, N, T, H)  # [B, N, T, H]

        # Create image token mask
        mask = inputs["image_embeds_position_mask"].view(B, N, T).bool()  # [B, N, T]

        # Extract image token features which have been language-conditioned by LLM
        img_emb = last_hidden_state[mask].reshape(B, N, -1, H)  # [B, N, T_img, H] where T_img = num image tokens

        # Extract text token features
        txt_emb = last_hidden_state[~mask].reshape(B, N, -1, H)  # [B, N, T_txt, H] where T_txt = num text tokens

        outputs = {}
        outputs["img_emb"] = img_emb
        outputs["txt_emb"] = txt_emb
        return outputs
    

class ContextEmbeddings(nn.Module):
    """Camera id and token type context embeddings
    
    Args:
        odim (int):
            The embedding size of processed observations (e.g., qpos, vision/language tokens)
        n_cams (int):
            The number of camera ids to generate embeddings for.

    Outputs:
        context_embeds (torch.Tensor):

    """
    def __init__(self, odim, n_cams):
        super().__init__()
        self.cam_id_emb = nn.Embedding(n_cams + 1, odim)
        self.tok_type_emb = nn.Embedding(3, odim)

        # Gate context embeddings for training stability (zero-initialized)
        # TODO: ensure that gates params are updated during training and don't remain zero
        self.g_cam_id = nn.Parameter(torch.zeros(1))
        self.g_tok_type = nn.Parameter(torch.zeros(1))

    def forward(self, tok_cam_ids, tok_type_ids, dtype):
        cam_embeds = self.cam_id_emb(tok_cam_ids) * self.g_cam_id
        type_embeds = self.tok_type_emb(tok_type_ids) * self.g_tok_type
        context_embeds = cam_embeds + type_embeds

        return context_embeds.to(dtype)



class Kosmos2DITPolicy(nn.Module):
    """VLA policy with Kosmos2 VLM backbone and DiT Block action head.
    Action head conditioned on VLM hidden_layer outputs via AdaLN-Zero mechanism.

    Arguments:
        lr (float):
            Learning rate for Adam.
        weight_decay (float):
            Regularization term for Adam optimizer.
        camera_names (Sequence[str]):
            The camera names.
        state_dim (int):
            The dimensionality in observations (e.g. qpos)
        action_dim (int):
            The dimensionality of the action space.
        prediction_horizon (int):
            The number of time steps we predict into the future (aka action chunk size)
        dim_feedforward (int):
            The dimensionality of the feedforward network in the action head.
        pool_text (bool):
            Whether to pool text embeddings output by the VL backbone per camera using a learnable weighted avg pooling method.
            Pooling is performed to reduce the number of tokens, lowering the O(S^2) cost of self-attention in the action head
            encoder. TODO - ABLATION STUDY RECOMMENDED TO EXPLORE EFFECT OF POOLING

    Members:
        vl_backbone (nn.Module): 
            The vision-language backbone from pretrained Kosmos2 checkpoint
        noise_net (nn.Module):
            The action head implemented as diffusion transformer with AdaLN-Zero conditioning
        n_cams (int):
            Number of cameras, we assume that cameras are named `cam1, cam2, ...`.
        state_dim (int):
            The number of observed action states (i.e., DoF)
        action_dim (int):
            The number of output action states (i.e., DoF)
        odim (int):
            The embedding size of all processed observations (e.g., qpos, vision/language tokens) 
            set to the dimensionality of the VLM token embeddings from the final hidden layer of
            a pretrained model checkpoint
        ac_chunk (int):
            The action chunk size (i.e., prediction horizon).
        train_diffusion_steps (int):
            The number of diffusion steps during training
        eval_diffusion_steps (int):
            The number of diffusion steps during evaluation
        pool_text_emb (bool):
            Whether to pool text embeddings output by the VL backbone per camera.
    """
    def __init__(
        self,
        *,
        lr,
        weight_decay,
        camera_names,
        state_dim,
        action_dim,
        prediction_horizon,
        dim_feedforward,
        pool_text=True, # TODO - ABLATION STUDY RECOMMENDED TO EXPLORE EFFECT OF POOLING
        **kwargs,
    ):
        super().__init__()

        # Set dimensions and chunk sizes
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._ac_chunk = prediction_horizon
        self._dim_feedforward = int(dim_feedforward)
        self._camera_names = camera_names
        self._n_cams = len(camera_names)

        # Instantiate VLM Backbone
        self.vl_backbone = Kosmos2Backbone(self.n_cams)
        self._odim = self.vl_backbone.model.config.text_config.embed_dim  # Expect 2048 for pretrained Kosmos2 checkpoint

        # Instantiate observation projection layer
        self.qpos_proj = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(self.state_dim, self.odim)
            )

        # Instantiate the action head
        action_head_kwargs = {
            'time_dim': 256,
            'hidden_dim': self.odim,
            'num_blocks': 6,
            'dim_feedforward': self._dim_feedforward,
            'dropout': 0.1,
            'nhead': 8,
        }
        self.noise_net = _DiTNoiseNet(
            action_dim=self.action_dim,
            ac_chunk=self.ac_chunk,
            **action_head_kwargs,
        )

        # DDIM requires fewer steps during evaluation than training
        train_diffusion_steps = 100
        eval_diffusion_steps = 8
        assert (
            eval_diffusion_steps <= train_diffusion_steps
        ), "Can't eval with more steps!"
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        
        # Instantiate diffusion scheduler
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        # Optimizer hyperparameters
        self._lr = lr
        self._weight_decay = weight_decay

        # Text pooling
        self._pool_text_emb = pool_text
        self._text_pool_proj = nn.Linear(self.odim, 1)

        # Context embeddings
        self._context_emb = ContextEmbeddings(self.odim, self.n_cams)

        # TODO
        if kwargs.get("enable_half_precision", False):
            raise NotImplementedError("Half precision not yet supported for Kosmos2DITPolicy")
        

    def __call__(self, qpos, images, actions, is_pad, text_command=None):
        """Forward inference during training. TODO - enable caching of vision-language embeddings for dataset items to speed up training

        Arguments:
            qpos (torch.Tensor):
                The state observations to tokenize alongside the images. 
                Tensor shape [B, `odim`]
            images (list[torch.Tensor], dtype=torch.uint8):
                A list of tensors of shape [C, H, W] * (B*N) where N is the number of cameras per 
                batch item, containing the visual observations to be tokenized.
            actions (torch.Tensor):
                A tensor of shape [B, K, A] where K is the action chunk size and A is the action
                dimension, containing the target actions.
            is_pad (torch.Tensor):
                A tensor of shape [B, K] where K is the action chunk size, containing a boolean mask
                indicating which actions to include in the loss. Generated for consistent padding in 
                cases where the dataloader returns the end of an episode.
            text_command (str):
                A list of text commands string with prefix/suffix to input into the VLM backbone.
                Shape: [N] * B
        """
        assert qpos.shape[1] == self.state_dim
        B, device = qpos.shape[0], qpos.device
        
        assert images[0].dtype == torch.uint8
        assert len(images) == B * self.n_cams
        assert actions.shape == (B, self.ac_chunk, self.action_dim), f"action.shape was {actions.shape}"        
        assert is_pad.shape == (B, self.ac_chunk)
        if text_command == None:
            import warnings
            warnings.warn("No text command provided to Kosmos2DITPolicy")

        # Encode all observations (vision, language instruction, proprioceptive state)
        vl_out = self.vl_backbone(images, text_command)  # dict[torch.Tensor]
        qpos_emb = self.qpos_proj(qpos)
        obs_out, _ = self._merge_observations(vl_out, qpos_emb)

        # Sample diffusion timestep for noising processs
        timesteps = torch.randint(
            low=0, high=self._train_diffusion_steps, size=(B,), device=device
        ).long()

        # TODO - DEBUG FORWARD PASS THROUGH NOISE_NET (dit_noise_net.py), ENSURING EXPECTED SHAPE + DIMENSIONALITY ALIGNMENT
        # Sample random noise and construct noised target actions with diffusion schedule
        noise = torch.randn_like(actions)
        noise_acs = self.diffusion_schedule.add_noise(actions, noise, timesteps)
        _, noise_pred = self.noise_net(noise_acs, timesteps, obs_out)

        # Calculate loss for noise net - # TODO: test noise_prediction loss vs. denoised action loss
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        mask = (~is_pad).unsqueeze(-1).float()           # [B, K, 1]
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return {"loss": loss}

    def get_actions(self, qpos, images, text_command=None, n_steps=None):
        """Forward inference during action prediction. 
        TODO - DEBUG inference time forward passes
        """
        assert qpos.shape[1] == self.state_dim
        B, device = qpos.shape[0], qpos.device
        
        assert images[0].dtype == torch.uint8
        assert len(images) == B * self.n_cams
        if text_command == None:
            import warnings
            warnings.warn("No text command provided to Kosmos2DITPolicy")

        # Encode all observations (vision, language instruction, proprioceptive state)
        vl_out = self.vl_backbone(images, text_command)  # dict[torch.Tensor]
        qpos_emb = self.qpos_proj(qpos)
        obs_out, _ = self._merge_observations(vl_out, qpos_emb)

        enc_cache = None
        noise_actions = torch.randn(B, self.ac_chunk, self.action_dim, device=device)

        # set number of steps
        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert (
                n_steps <= self._train_diffusion_steps
            ), f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps

        enc_cache = self.noise_net.forward_enc(obs_out)

        # begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = (
            self.diffusion_schedule.alphas_cumprod.to(device)
        )
        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net.forward_dec(noise_actions, batched_timestep, enc_cache)

            # take diffusion step
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=timestep, sample=noise_actions
            ).prev_sample

        # return final action post diffusion
        assert noise_actions.shape == (B, self.ac_chunk, self.action_dim)
        return noise_actions
    
    def _merge_observations(self, vl_embeds, qpos_emb):
        """
        Merges vision-language embeddings and proprioceptive state (qpos) embedding into a joint sequence 
        of observation embeddings to be passed to the DiT Block encoder.

        Args:
            vl_embeds (dict[torch.Tensor]): Vision language embeddings
                "img_emb" : image embeddings [B, N, T_img, H]
                "txt_emb" : text embeddings [B, N, T_txt, H]
            qpos_emb (torch.Tensor): Proprioceptive state (qpos) embedding [B, H]

        Outputs:
            obs_out (torch.Tensor): Merged observation embeddings flattened along the sequence dimension [B, S, H]
            context_embeds (torch.Tensor): Context embeddings for the merged observation embeddings [B, S, H]
        """
        img = vl_embeds["img_emb"]
        txt = vl_embeds["txt_emb"]
        B, N, T_img, H = img.shape
        T_txt = txt.size(2)
        
        # Optionally pool text embeddings
        # Note: Pooling is performed to reduce the number of tokens, lowering the O(S^2) cost of self-attention in the action head encoder.
        #       We expect that language conditioning information to be largely preserved in per-cam visual tokens thanks to cross-attention 
        #       in the VLM backbone.
        
        if self.pool_text_emb:
            # Pool text embeddings using a learnable weighted avg pooling method
            w = self._text_pool_proj(txt)  # [B, N, T_txt, 1]
            score = torch.softmax(w, dim=2)  # [B, N, T_txt, 1]
            txt = (score * txt).sum(dim=2, keepdim=True)  # pooled text embedding per camera view [B, N, 1, H]

        T_txt_remaining = txt.shape[2]

        # Flatten embeddings along the token sequence dim and concatenate
        img = img.reshape(B, N * T_img, H)
        txt = txt.reshape(B, N * T_txt_remaining, H)
        qpos = qpos_emb.unsqueeze(1)

        obs = torch.cat([img, txt, qpos], dim=1)  # [B, S, H]
        S = obs.size(1)
        device = obs.device

        # Create camera type embedding ids.
        img_cam_ids = torch.arange(N, dtype=torch.long, device=device).repeat_interleave(T_img)
        img_cam_ids = img_cam_ids.unsqueeze(0).repeat(B, 1)
        txt_cam_ids = torch.arange(N, dtype=torch.long, device=device).repeat_interleave(T_txt_remaining)
        txt_cam_ids = txt_cam_ids.unsqueeze(0).repeat(B, 1)
        qpos_cam_ids = torch.full((B, 1), N, dtype=torch.long, device=device)  # assigns special no-cam id

        tok_cam_ids = torch.cat([img_cam_ids, txt_cam_ids, qpos_cam_ids], dim=1)  # [B, S]
        
        # Create token type embedding ids
        #   img_tok_id = 0
        #   txt_tok_id = 1
        #   qpos_tok_id = 2
        img_type_ids = torch.zeros(B, N * T_img, dtype=torch.long, device=device)
        txt_type_ids = torch.ones(B, N * T_txt_remaining, dtype=torch.long, device=device)
        qpos_type_ids = torch.full((B, 1), 2, dtype=torch.long, device=device)

        tok_type_ids = torch.cat([img_type_ids, txt_type_ids, qpos_type_ids], dim=1)  # [B, S]

        # Apply the combined context embeddings for the camera and token types
        context_embeds = self._context_emb(tok_cam_ids, tok_type_ids, dtype=obs.dtype)  # [B, S, H]
        obs_out = obs + context_embeds  # [B, S, H]

        return obs_out, context_embeds


    @property
    def odim(self)-> int:
        return self._odim
    
    @property
    def n_cams(self) -> int:
        return self._n_cams
    
    @property
    def ac_chunk(self) -> int:
        return self._ac_chunk

    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def train_diffusion_steps(self) -> int:
        return self._train_diffusion_steps

    @property
    def eval_diffusion_steps(self) -> int:
        return self._eval_diffusion_steps

    @property
    def pool_text_emb(self) -> bool:
        return self._pool_text_emb

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr, 
                                      weight_decay=self._weight_decay, 
                                      betas=[0.95, 0.999], eps=1.0e-8)
        return optimizer
    
    def configure_scheduler(self, optimizer: torch.optim.Optimizer, num_steps: int):
        schedule = diffusers.optimization.get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                                          num_warmup_steps=2000, 
                                                                          num_training_steps=num_steps)
        return schedule
    
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)