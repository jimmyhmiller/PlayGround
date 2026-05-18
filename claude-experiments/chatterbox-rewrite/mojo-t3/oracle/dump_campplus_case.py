"""
Dump CAMPPlus inputs + outputs for parity testing the pure-Mojo port.

Captures:
  ref_wav_16k.bin    (T_samples,)         16kHz mono audio
  fbank_feat.bin     (T_frames, 80)        Kaldi fbank features
  fbank_mean.bin     (80,)                 per-utterance mean (for centering)
  xvector.bin        (1, 192)              CAMPPlus output (the cloned-voice embedding)
  weights/*.bin      flat dump of every parameter
  weights_manifest.txt
"""
from __future__ import annotations
import os, sys, struct
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "campplus"
OUT.mkdir(parents=True, exist_ok=True)
WDIR = OUT / "weights"
WDIR.mkdir(exist_ok=True)

REF_WAV = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"


def write_tensor(p, arr):
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
    else:
        raise TypeError(arr.dtype)
    with p.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    model = ChatterboxTTS.from_pretrained(device=device)
    spk = model.s3gen.speaker_encoder

    # Strip weight_norm if any (CAMPPlus mostly doesn't have it but be safe).
    import torch.nn.utils.parametrize as parametrize
    def strip(m):
        if hasattr(m, "parametrizations"):
            for n in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, n, leave_parametrized=True)
        for c in m.children():
            strip(c)
    strip(spk)

    # Load reference WAV at 16kHz mono.
    import wave
    with wave.open(REF_WAV, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        nc = wf.getnchannels()
        raw = wf.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, nc).mean(axis=1).astype(np.float32) / 32768.0
    if sr != 16000:
        import scipy.signal as sig
        pcm = sig.resample_poly(pcm, 16000, sr).astype(np.float32)
    # Truncate to 10s for fixture size.
    pcm = pcm[: 16000 * 10]
    wav = torch.from_numpy(pcm).unsqueeze(0).to(device)
    print(f"[campplus] wav: {tuple(wav.shape)} sr=16000")
    write_tensor(OUT / "ref_wav_16k.bin", wav.cpu().numpy().astype(np.float32))

    # Capture the Kaldi fbank features that extract_feature() produces.
    from chatterbox.models.s3gen.xvector import extract_feature
    feats, lens, times = extract_feature(wav)
    # feats is (B, T_frames, 80), but extract_feature centers per-utterance.
    print(f"[campplus] fbank: {tuple(feats.shape)}")
    write_tensor(OUT / "fbank_feat.bin", feats.cpu().numpy().astype(np.float32))

    # Capture the FCM head's first-layer (conv1 -> bn1 -> relu) output.
    head = spk.head
    fcm_input = feats.to(device).to(torch.float32).permute(0, 2, 1)  # (B,80,T)
    fcm_input_4d = fcm_input.unsqueeze(1)  # (B,1,80,T)
    with torch.inference_mode():
        conv1_out = head.conv1(fcm_input_4d)
        bn1_out = head.bn1(conv1_out)
        relu1_out = torch.relu(bn1_out)
    write_tensor(OUT / "fcm_input_4d.bin", fcm_input_4d.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "fcm_conv1_out.bin", conv1_out.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "fcm_relu1_out.bin", relu1_out.cpu().numpy().astype(np.float32))

    # Capture each BasicResBlock output along the FCM path.
    with torch.inference_mode():
        # Block-by-block inside layer1 / layer2 so the Mojo test can compare
        # at a fine grain. layer1 has 2 BasicResBlocks; first has stride=2.
        x_l1_0 = head.layer1[0](relu1_out)
        x_l1 = head.layer1[1](x_l1_0)
        write_tensor(OUT / "fcm_layer1_block0_out.bin",
                     x_l1_0.cpu().numpy().astype(np.float32))
        x_l2 = head.layer2(x_l1)             # (B, 32, 20, 998) — stride=2 on H
        # head.conv2 has stride=(2,1), padding=1 → H/2
        x_c2 = head.conv2(x_l2)              # (B, 32, 10, 998)
        x_bn2 = head.bn2(x_c2)
        x_relu2 = torch.relu(x_bn2)
        # Final FCM output: flatten (C, H) → (C*H, T): (1, 320, 998)
        x_fcm = x_relu2.reshape(x_relu2.shape[0], x_relu2.shape[1] * x_relu2.shape[2], x_relu2.shape[3])
    write_tensor(OUT / "fcm_layer1_out.bin", x_l1.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "fcm_layer2_out.bin", x_l2.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "fcm_out.bin", x_fcm.cpu().numpy().astype(np.float32))

    # TDNN trunk intermediates.
    xv = spk.xvector
    x = x_fcm
    with torch.inference_mode():
        x_tdnn = xv.tdnn(x)
        # First dense TDNN layer (block1.tdnnd1) and its internal stages.
        l = xv.block1.tdnnd1
        nl1_out = l.nonlinear1(x_tdnn)
        lin1_out = l.linear1(nl1_out)
        nl2_out = l.nonlinear2(lin1_out)
        # CAMLayer internals.
        cam = l.cam_layer
        y_local = cam.linear_local(nl2_out)
        ctx_mean = nl2_out.mean(-1, keepdim=True)
        ctx_seg = cam.seg_pooling(nl2_out)
        ctx_sum = ctx_mean + ctx_seg
        ctx_l1 = cam.linear1(ctx_sum)
        ctx_relu = torch.relu(ctx_l1)
        ctx_l2 = cam.linear2(ctx_relu)
        m = torch.sigmoid(ctx_l2)
        cam_out = y_local * m
        # After cam, the dense layer concatenates [x_tdnn, cam_out].
        tdnnd1_out = torch.cat([x_tdnn, cam_out], dim=1)
        write_tensor(OUT / "tdnnd1_nl1.bin", nl1_out.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_lin1.bin", lin1_out.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_nl2.bin", nl2_out.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_cam_y_local.bin", y_local.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_cam_ctx_mean.bin", ctx_mean.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_cam_ctx_seg.bin", ctx_seg.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_cam_m.bin", m.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_cam_out.bin", cam_out.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "tdnnd1_out.bin", tdnnd1_out.cpu().numpy().astype(np.float32))
        print(f"[tdnnd1] nl1:{tuple(nl1_out.shape)} lin1:{tuple(lin1_out.shape)} "
              f"nl2:{tuple(nl2_out.shape)} y_local:{tuple(y_local.shape)} "
              f"ctx_mean:{tuple(ctx_mean.shape)} ctx_seg:{tuple(ctx_seg.shape)} "
              f"m:{tuple(m.shape)} cam_out:{tuple(cam_out.shape)} "
              f"tdnnd1_out:{tuple(tdnnd1_out.shape)}")
        x_b1 = xv.block1(x_tdnn)
        x_t1 = xv.transit1(x_b1)
        x_b2 = xv.block2(x_t1)
        x_t2 = xv.transit2(x_b2)
        x_b3 = xv.block3(x_t2)
        x_t3 = xv.transit3(x_b3)
        x_out = xv.out_nonlinear(x_t3)
        x_stats = xv.stats(x_out)
        x_dense = xv.dense(x_stats)
    write_tensor(OUT / "tdnn_out.bin",       x_tdnn.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "block1_out.bin",     x_b1.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "transit1_out.bin",   x_t1.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "block2_out.bin",     x_b2.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "transit2_out.bin",   x_t2.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "block3_out.bin",     x_b3.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "transit3_out.bin",   x_t3.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "out_nonlinear.bin",  x_out.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "stats_out.bin",      x_stats.cpu().numpy().astype(np.float32))
    write_tensor(OUT / "dense_out.bin",      x_dense.cpu().numpy().astype(np.float32))
    print(f"[campplus] tdnn:{tuple(x_tdnn.shape)} b1:{tuple(x_b1.shape)} t1:{tuple(x_t1.shape)} "
          f"b2:{tuple(x_b2.shape)} t2:{tuple(x_t2.shape)} b3:{tuple(x_b3.shape)} t3:{tuple(x_t3.shape)} "
          f"on:{tuple(x_out.shape)} stats:{tuple(x_stats.shape)} dense:{tuple(x_dense.shape)}")

    # Run CAMPPlus.
    with torch.inference_mode():
        xvec = spk(feats.to(device).to(torch.float32))
    print(f"[campplus] xvector: {tuple(xvec.shape)} (first 8): {xvec.flatten()[:8].cpu().numpy()}")
    write_tensor(OUT / "xvector.bin", xvec.cpu().numpy().astype(np.float32))

    # Also try CAMPPlus.inference() to be the same path s3gen uses.
    with torch.inference_mode():
        xvec_inf = spk.inference([wav.squeeze(0)])
    print(f"[campplus] inference xvector: {tuple(xvec_inf.shape)} first 8: {xvec_inf.flatten()[:8].cpu().numpy()}")
    write_tensor(OUT / "xvector_inference.bin", xvec_inf.cpu().numpy().astype(np.float32))

    # Dump every state_dict tensor.
    sd = spk.state_dict()
    print(f"[campplus] {len(sd)} weights")
    for k, v in sd.items():
        fname = k.replace(".", "__") + ".bin"
        write_tensor(WDIR / fname, v.detach().cpu().numpy().astype(np.float32))
    with (OUT / "weights_manifest.txt").open("w") as f:
        for k in sorted(sd.keys()):
            f.write(f"{k}\t{tuple(sd[k].shape)}\n")


if __name__ == "__main__":
    main()
