"""Run upstream's flow encoder on Mojo's bad tokens → dump mu, spks, cond, etc.
Save to /tmp/cfm_diag/upstream_mu_on_mojo_tokens.npz so Mojo can compare.
"""
import sys, importlib.metadata as _im, types
sys.path.insert(0, '/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src')
_orig = _im.version
def _v(name):
    try: return _orig(name)
    except _im.PackageNotFoundError: return "0.0.0"
_im.version = _v

import numpy as np, torch
from chatterbox.tts import ChatterboxTTS as UpstreamTTS

REF = '/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav'
mojo_tokens = np.load('/tmp/cfm_diag/mojo_bad_tokens.npy').tolist()
print(f"loaded mojo tokens: {len(mojo_tokens)}", flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ups = UpstreamTTS.from_pretrained(device=device)
ups.prepare_conditionals(REF, exaggeration=0.5)

# Hook the CFM forward to capture mu/cond/spks/mask
captured = {}
cfm = ups.s3gen.flow.decoder

@torch.inference_mode()
def patched_cfm_forward(self_cfm, mu, mask, n_timesteps, temperature=1.0,
                        spks=None, cond=None, noised_mels=None, meanflow=False):
    captured["mu"] = mu.detach().cpu().float().numpy().copy()
    captured["mask"] = mask.detach().cpu().float().numpy().copy()
    captured["spks"] = spks.detach().cpu().float().numpy().copy() if spks is not None else None
    captured["cond"] = cond.detach().cpu().float().numpy().copy() if cond is not None else None
    z = torch.randn_like(mu)
    if noised_mels is not None:
        prompt_len = mu.size(2) - noised_mels.size(2)
        z = z.clone()
        z[..., prompt_len:] = noised_mels
    captured["z_init"] = z.detach().cpu().float().numpy().copy()
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    if (not meanflow) and (self_cfm.t_scheduler == "cosine"):
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    captured["t_span"] = t_span.detach().cpu().float().numpy().copy()
    x = self_cfm.solve_euler(z, t_span=t_span, mu=mu, mask=mask,
                              spks=spks, cond=cond, meanflow=meanflow)
    captured["final_x"] = x.detach().cpu().float().numpy().copy()
    return x, None
cfm.forward = types.MethodType(patched_cfm_forward, cfm)

speech_tokens = torch.tensor(mojo_tokens, dtype=torch.long, device=device).unsqueeze(0)
with torch.inference_mode():
    wav, _ = ups.s3gen.inference(speech_tokens=speech_tokens, ref_dict=ups.conds.gen)

print(f"mu: {captured['mu'].shape}, spks: {captured['spks'].shape if captured['spks'] is not None else None}, cond: {captured['cond'].shape}", flush=True)
np.savez("/tmp/cfm_diag/upstream_on_mojo.npz",
         mu=captured["mu"], mask=captured["mask"], spks=captured["spks"],
         cond=captured["cond"], z_init=captured["z_init"], t_span=captured["t_span"],
         final_x=captured["final_x"])
print("saved /tmp/cfm_diag/upstream_on_mojo.npz", flush=True)
