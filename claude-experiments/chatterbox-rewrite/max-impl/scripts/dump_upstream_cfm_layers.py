"""Dump upstream's CFM ConditionalDecoder forward pass — per-layer outputs.

Hooks every key tensor in the forward pass for one chunk's CFM call (chunk 28
of Warrant Chapter 0, using Mojo bf16-T3 tokens).

Saves to /tmp/cfm_diag/upstream_layers.npz.
"""
import sys, importlib.metadata as _im, types
sys.path.insert(0, '/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src')
_orig = _im.version
def _v(n):
    try: return _orig(n)
    except _im.PackageNotFoundError: return '0.0.0'
_im.version = _v

import torch, numpy as np
torch.manual_seed(0)  # fix CFM noise

from chatterbox.tts import ChatterboxTTS

REF = '/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav'
tokens = np.load('/tmp/cfm_diag/warrant28_mojo_bf16_tokens.npy').tolist()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ups = ChatterboxTTS.from_pretrained(device=device)
ups.prepare_conditionals(REF, exaggeration=0.5)

captured = {}

def _save(name, t):
    captured[name] = t.detach().cpu().float().numpy().copy()

# Hook ConditionalDecoder.forward to instrument every layer.
cfm = ups.s3gen.flow.decoder.estimator

orig_forward = cfm.forward

@torch.inference_mode()
def patched_estimator_forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
    """Mirror upstream forward but save every intermediate."""
    # Only save on step 0 to keep output small.
    step_idx = captured.get('_step_call_count', 0)
    captured['_step_call_count'] = step_idx + 1

    save = (step_idx == 0)  # capture first solver step only

    t_emb_sin = self.time_embeddings(t).to(t.dtype)
    if save: _save('t_emb_sinusoidal', t_emb_sin)
    t_emb = self.time_mlp(t_emb_sin)
    if save: _save('t_emb_mlp', t_emb)

    # Pack inputs: [x, mu, spks_broadcast, cond] along channel dim.
    from einops import pack, rearrange, repeat
    x_packed = pack([x, mu], 'b * t')[0]
    if save: _save('x_after_pack_mu', x_packed)
    if spks is not None:
        spks_b = repeat(spks, 'b c -> b c t', t=x_packed.shape[-1])
        x_packed = pack([x_packed, spks_b], 'b * t')[0]
        if save: _save('x_after_pack_spks', x_packed)
    if cond is not None:
        x_packed = pack([x_packed, cond], 'b * t')[0]
        if save: _save('x_after_pack_cond', x_packed)

    x = x_packed
    hiddens = []
    masks = [mask]
    from chatterbox.models.s3gen.utils.mask import add_optional_chunk_mask
    from chatterbox.models.s3gen.decoder import mask_to_bias

    for di, (resnet, transformer_blocks, downsample) in enumerate(self.down_blocks):
        mask_down = masks[-1]
        x = resnet(x, mask_down, t_emb)
        if save: _save(f'down{di}_after_resnet', x)
        x = rearrange(x, 'b c t -> b t c').contiguous()
        attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
        attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
        for ti, tb in enumerate(transformer_blocks):
            x = tb(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
            if save: _save(f'down{di}_t{ti}', x)
        x = rearrange(x, 'b t c -> b c t').contiguous()
        hiddens.append(x)
        x = downsample(x * mask_down)
        if save: _save(f'down{di}_after_downsample', x)
        masks.append(mask_down[:, :, ::2])
    masks = masks[:-1]
    mask_mid = masks[-1]

    for mi, (resnet, transformer_blocks) in enumerate(self.mid_blocks):
        x = resnet(x, mask_mid, t_emb)
        if save: _save(f'mid{mi}_after_resnet', x)
        x = rearrange(x, 'b c t -> b t c').contiguous()
        attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
        attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
        for ti, tb in enumerate(transformer_blocks):
            x = tb(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
            if save: _save(f'mid{mi}_t{ti}', x)
        x = rearrange(x, 'b t c -> b c t').contiguous()
        if save: _save(f'mid{mi}_done', x)

    for ui, (resnet, transformer_blocks, upsample) in enumerate(self.up_blocks):
        mask_up = masks.pop()
        skip = hiddens.pop()
        x = pack([x[:, :, :skip.shape[-1]], skip], 'b * t')[0]
        if save: _save(f'up{ui}_after_skip', x)
        x = resnet(x, mask_up, t_emb)
        if save: _save(f'up{ui}_after_resnet', x)
        x = rearrange(x, 'b c t -> b t c').contiguous()
        attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
        attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
        for ti, tb in enumerate(transformer_blocks):
            x = tb(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
            if save: _save(f'up{ui}_t{ti}', x)
        x = rearrange(x, 'b t c -> b c t').contiguous()
        x = upsample(x * mask_up)
        if save: _save(f'up{ui}_after_upsample', x)
    x = self.final_block(x, mask_up)
    if save: _save('final_block', x)
    output = self.final_proj(x * mask_up)
    if save: _save('final_proj', output)
    return output

cfm.forward = types.MethodType(patched_estimator_forward, cfm)

# Also hook CFM solver to use deterministic noise.
solver = ups.s3gen.flow.decoder
orig_solver_forward = solver.forward

@torch.inference_mode()
def patched_solver_forward(self, mu, mask, n_timesteps, temperature=1.0,
                            spks=None, cond=None, noised_mels=None, meanflow=False):
    z = torch.randn_like(mu)
    captured['z_init'] = z.detach().cpu().float().numpy().copy()
    captured['mu_in'] = mu.detach().cpu().float().numpy().copy()
    captured['mask_in'] = mask.detach().cpu().float().numpy().copy()
    captured['spks_in'] = spks.detach().cpu().float().numpy().copy()
    captured['cond_in'] = cond.detach().cpu().float().numpy().copy()
    t_span = 1 - torch.cos(torch.linspace(0, 1, n_timesteps+1, device=mu.device, dtype=mu.dtype) * 0.5 * torch.pi)
    captured['t_span'] = t_span.detach().cpu().float().numpy().copy()
    x = self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, meanflow=False)
    captured['final_x'] = x.detach().cpu().float().numpy().copy()
    return x, None
solver.forward = types.MethodType(patched_solver_forward, solver)

with torch.inference_mode():
    ups.s3gen.inference(speech_tokens=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                        ref_dict=ups.conds.gen)

print(f"captured {len(captured)} tensors")
for k in sorted(captured.keys()):
    if k.startswith('_'): continue
    v = captured[k]
    print(f"  {k}: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}], rms={float(np.sqrt((v**2).mean())):.3f}")

np.savez('/tmp/cfm_diag/upstream_layers.npz', **{k: v for k, v in captured.items() if not k.startswith('_')})
print('saved /tmp/cfm_diag/upstream_layers.npz')
