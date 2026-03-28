export const Float32Array_ID = idof<Float32Array>();

function kernel_0(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[d2]);
    _out[oi] = t0;
  }
}

function kernel_1(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[(d0 * 128) + d1]);
    _out[oi] = t0;
  }
}

function kernel_2(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (-t1);
    const t3: f32 = (t0 + t2);
    const t4: f32 = unchecked(_in2[oi]);
    const t5: f32 = (t3 < t4 ? f32(1.0) : f32(0.0));
    _out[oi] = t5;
  }
}

function kernel_3(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = unchecked(_in2[0]);
    const t3: f32 = (t1 + t2);
    const t4: f32 = (t0 < t3 ? f32(1.0) : f32(0.0));
    _out[oi] = t4;
  }
}

function kernel_4(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 811008; oi++) {
    const d0: i32 = (oi / 811008) % 1;
    const d1: i32 = (oi / 6336) % 128;
    const d2: i32 = (oi / 64) % 99;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d2]);
    const t1: f32 = unchecked(_in1[(d0 * 12672) + (d1 * 99) + d2]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_5(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 811008; oi++) {
    const d0: i32 = (oi / 811008) % 1;
    const d1: i32 = (oi / 6336) % 128;
    const d2: i32 = (oi / 64) % 99;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d2 * 64) + d3]);
    _out[oi] = t0;
  }
}

function kernel_6(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 8192 + d1 * 64 + d2 * 64;
          for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
            const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 64) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_7(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d2]);
    const t1: f32 = unchecked(_in1[(d1 * 64) + d2]);
    const t2: f32 = (t0 + t1);
    _out[oi] = t2;
  }
}

function kernel_8(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 16384; oi++) {
    const d0: i32 = (oi / 16384) % 1;
    const d1: i32 = (oi / 16384) % 1;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[d2]);
    const t1: f32 = unchecked(_in1[d3]);
    const t2: f32 = (t0 < t1 ? f32(1.0) : f32(0.0));
    _out[oi] = t2;
  }
}

function kernel_9(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1; oi++) {
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = (-t0);
    _out[oi] = t1;
  }
}

function kernel_10(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d2 * 128) + d3]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_11(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
              const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
              const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_12(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 128) + d1]);
    const t2: f32 = unchecked(_in2[0]);
    const t3: f32 = (t1 * t2);
    const t4: f32 = (-t3);
    const t5: f32 = (t0 + t4);
    _out[oi] = t5;
  }
}

function kernel_13(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
              const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                const t1: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                const t2: f32 = (t0 * t1);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t2;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
              const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                const t1: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                const t2: f32 = (t0 * t1);
                _acc_r = _acc_r + t2;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_14(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 128; oi++) {
    const d0: i32 = (oi / 128) % 1;
    const d1: i32 = (oi / 1) % 128;
    const d2: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[0]);
    const t4: f32 = (t2 + t3);
    _out[oi] = t4;
  }
}

function kernel_15(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 128; oi++) {
    const d0: i32 = (oi / 128) % 1;
    const d1: i32 = (oi / 1) % 128;
    const d2: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = f32(Math.sqrt(f64(t0)));
    _out[oi] = t1;
  }
}

function kernel_16(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 128; oi++) {
    const d0: i32 = (oi / 128) % 1;
    const d1: i32 = (oi / 1) % 128;
    const d2: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = (f32(1.0) / t0);
    _out[oi] = t1;
  }
}

function kernel_17(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 128) + d1]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_18(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 1572864; oi++) {
    const d0: i32 = (oi / 1572864) % 1;
    const d1: i32 = (oi / 12288) % 128;
    const d2: i32 = (oi / 192) % 64;
    const d3: i32 = oi % 192;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d2]);
    const t1: f32 = unchecked(_in1[d2]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[d2]);
    const t4: f32 = (t2 + t3);
    _out[oi] = t4;
  }
}

function kernel_19(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1572864; oi++) {
    const d0: i32 = (oi / 1572864) % 1;
    const d1: i32 = (oi / 12288) % 128;
    const d2: i32 = (oi / 192) % 64;
    const d3: i32 = oi % 192;
    const t0: f32 = unchecked(_in0[(d2 * 192) + d3]);
    _out[oi] = t0;
  }
}

function kernel_20(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 24576 + d1 * 192 + d2 * 192;
          for (let n_blk: i32 = 0; n_blk < 6; n_blk++) {
            const n_tile: i32 = 192 - n_blk * 32 < 32 ? 192 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d4 * 192) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_21(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 24576; oi++) {
    const d0: i32 = (oi / 24576) % 1;
    const d1: i32 = (oi / 192) % 128;
    const d2: i32 = oi % 192;
    const t0: f32 = unchecked(_in0[(d0 * 24576) + (d1 * 192) + d2]);
    const t1: f32 = unchecked(_in1[d2]);
    const t2: f32 = (t0 + t1);
    _out[oi] = t2;
  }
}

function kernel_22(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 24576; oi++) {
    const d0: i32 = (oi / 24576) % 1;
    const d1: i32 = (oi / 192) % 128;
    const d2: i32 = (oi / 64) % 3;
    const d3: i32 = (oi / 16) % 4;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 24576) + (d1 * 192) + (d2 * 64) + (d3 * 16) + d4]);
    _out[oi] = t0;
  }
}

function kernel_23(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = (oi / 64) % 1;
    const d3: i32 = (oi / 16) % 4;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 24576) + (d1 * 192) + (d2 * 64) + (d3 * 16) + d4]);
    _out[oi] = t0;
  }
}

function kernel_24(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = (oi / 64) % 1;
    const d3: i32 = (oi / 16) % 4;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 24576) + (d1 * 192) + (d2 * 64) + (d3 * 16) + d4 + 64]);
    _out[oi] = t0;
  }
}

function kernel_25(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = (oi / 64) % 1;
    const d3: i32 = (oi / 16) % 4;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 24576) + (d1 * 192) + (d2 * 64) + (d3 * 16) + d4 + 128]);
    _out[oi] = t0;
  }
}

function kernel_26(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1048576; oi++) {
    const d0: i32 = (oi / 1048576) % 1;
    const d1: i32 = (oi / 262144) % 4;
    const d2: i32 = (oi / 2048) % 128;
    const d3: i32 = (oi / 128) % 16;
    const d4: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d2 * 64) + (d1 * 16) + d3]);
    _out[oi] = t0;
  }
}

function kernel_27(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1048576; oi++) {
    const d0: i32 = (oi / 1048576) % 1;
    const d1: i32 = (oi / 262144) % 4;
    const d2: i32 = (oi / 2048) % 128;
    const d3: i32 = (oi / 128) % 16;
    const d4: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d4 * 64) + (d1 * 16) + d3]);
    _out[oi] = t0;
  }
}

function kernel_28(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let d2: i32 = 0; d2 < 128; d2++) {
        for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
          const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
          for (let mi: i32 = 0; mi < m_end; mi++) {
            const d3: i32 = m_blk * 1 + mi;
            const _row_base: i32 = d0 * 65536 + d1 * 16384 + d2 * 128 + d3 * 128;
            for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
              const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
              for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
                const ni_base: i32 = n_blk * 32 + ni_grp * 32;
                let _acc0: f32 = f32(0.0);
                let _acc1: f32 = f32(0.0);
                let _acc2: f32 = f32(0.0);
                let _acc3: f32 = f32(0.0);
                let _acc4: f32 = f32(0.0);
                let _acc5: f32 = f32(0.0);
                let _acc6: f32 = f32(0.0);
                let _acc7: f32 = f32(0.0);
                let _acc8: f32 = f32(0.0);
                let _acc9: f32 = f32(0.0);
                let _acc10: f32 = f32(0.0);
                let _acc11: f32 = f32(0.0);
                let _acc12: f32 = f32(0.0);
                let _acc13: f32 = f32(0.0);
                let _acc14: f32 = f32(0.0);
                let _acc15: f32 = f32(0.0);
                let _acc16: f32 = f32(0.0);
                let _acc17: f32 = f32(0.0);
                let _acc18: f32 = f32(0.0);
                let _acc19: f32 = f32(0.0);
                let _acc20: f32 = f32(0.0);
                let _acc21: f32 = f32(0.0);
                let _acc22: f32 = f32(0.0);
                let _acc23: f32 = f32(0.0);
                let _acc24: f32 = f32(0.0);
                let _acc25: f32 = f32(0.0);
                let _acc26: f32 = f32(0.0);
                let _acc27: f32 = f32(0.0);
                let _acc28: f32 = f32(0.0);
                let _acc29: f32 = f32(0.0);
                let _acc30: f32 = f32(0.0);
                let _acc31: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                  const k_end: i32 = 16 - k_blk * 16 < 16 ? 16 - k_blk * 16 : 16;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 16 + ki;
                    {
                      const d4: i32 = ni_base + 0;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc0 = _acc0 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 1;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc1 = _acc1 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 2;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc2 = _acc2 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 3;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc3 = _acc3 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 4;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc4 = _acc4 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 5;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc5 = _acc5 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 6;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc6 = _acc6 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 7;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc7 = _acc7 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 8;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc8 = _acc8 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 9;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc9 = _acc9 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 10;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc10 = _acc10 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 11;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc11 = _acc11 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 12;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc12 = _acc12 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 13;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc13 = _acc13 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 14;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc14 = _acc14 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 15;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc15 = _acc15 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 16;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc16 = _acc16 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 17;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc17 = _acc17 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 18;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc18 = _acc18 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 19;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc19 = _acc19 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 20;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc20 = _acc20 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 21;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc21 = _acc21 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 22;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc22 = _acc22 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 23;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc23 = _acc23 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 24;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc24 = _acc24 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 25;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc25 = _acc25 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 26;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc26 = _acc26 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 27;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc27 = _acc27 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 28;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc28 = _acc28 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 29;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc29 = _acc29 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 30;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc30 = _acc30 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 31;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc31 = _acc31 + t2;
                    }
                  }
                }
                unchecked(_out[_row_base + ni_base + 0] = _acc0);
                unchecked(_out[_row_base + ni_base + 1] = _acc1);
                unchecked(_out[_row_base + ni_base + 2] = _acc2);
                unchecked(_out[_row_base + ni_base + 3] = _acc3);
                unchecked(_out[_row_base + ni_base + 4] = _acc4);
                unchecked(_out[_row_base + ni_base + 5] = _acc5);
                unchecked(_out[_row_base + ni_base + 6] = _acc6);
                unchecked(_out[_row_base + ni_base + 7] = _acc7);
                unchecked(_out[_row_base + ni_base + 8] = _acc8);
                unchecked(_out[_row_base + ni_base + 9] = _acc9);
                unchecked(_out[_row_base + ni_base + 10] = _acc10);
                unchecked(_out[_row_base + ni_base + 11] = _acc11);
                unchecked(_out[_row_base + ni_base + 12] = _acc12);
                unchecked(_out[_row_base + ni_base + 13] = _acc13);
                unchecked(_out[_row_base + ni_base + 14] = _acc14);
                unchecked(_out[_row_base + ni_base + 15] = _acc15);
                unchecked(_out[_row_base + ni_base + 16] = _acc16);
                unchecked(_out[_row_base + ni_base + 17] = _acc17);
                unchecked(_out[_row_base + ni_base + 18] = _acc18);
                unchecked(_out[_row_base + ni_base + 19] = _acc19);
                unchecked(_out[_row_base + ni_base + 20] = _acc20);
                unchecked(_out[_row_base + ni_base + 21] = _acc21);
                unchecked(_out[_row_base + ni_base + 22] = _acc22);
                unchecked(_out[_row_base + ni_base + 23] = _acc23);
                unchecked(_out[_row_base + ni_base + 24] = _acc24);
                unchecked(_out[_row_base + ni_base + 25] = _acc25);
                unchecked(_out[_row_base + ni_base + 26] = _acc26);
                unchecked(_out[_row_base + ni_base + 27] = _acc27);
                unchecked(_out[_row_base + ni_base + 28] = _acc28);
                unchecked(_out[_row_base + ni_base + 29] = _acc29);
                unchecked(_out[_row_base + ni_base + 30] = _acc30);
                unchecked(_out[_row_base + ni_base + 31] = _acc31);
              }
              for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
                const d4: i32 = n_blk * 32 + ni;
                let _acc_r: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                  const k_end: i32 = 16 - k_blk * 16 < 16 ? 16 - k_blk * 16 : 16;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 16 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 128) + d4]);
                    const t2: f32 = (t0 * t1);
                    _acc_r = _acc_r + t2;
                  }
                }
                unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
              }
            }
          }
        }
      }
    }
  }
}

function kernel_29(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
    _out[oi] = t0;
  }
}

function kernel_30(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[oi]);
    const t4: f32 = (t2 + t3);
    _out[oi] = t4;
  }
}

function kernel_31(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 512 + d1 * 128 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(-Infinity);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = t0 > _acc0 ? t0 : _acc0;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(-Infinity);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  _acc_r = t0 > _acc_r ? t0 : _acc_r;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_32(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 512) + (d1 * 128) + d2]);
    const t2: f32 = (-t1);
    const t3: f32 = (t0 + t2);
    _out[oi] = t3;
  }
}

function kernel_33(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_34(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = f32(Math.pow(2.0, f64(t0)));
    _out[oi] = t1;
  }
}

function kernel_35(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 512 + d1 * 128 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t0;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  _acc_r = _acc_r + t0;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_36(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 512; oi++) {
    const d0: i32 = (oi / 512) % 1;
    const d1: i32 = (oi / 128) % 4;
    const d2: i32 = (oi / 1) % 128;
    const d3: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = (f32(1.0) / t0);
    _out[oi] = t1;
  }
}

function kernel_37(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 1048576; oi++) {
    const d0: i32 = (oi / 1048576) % 1;
    const d1: i32 = (oi / 262144) % 4;
    const d2: i32 = (oi / 2048) % 128;
    const d3: i32 = (oi / 16) % 128;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 512) + (d1 * 128) + d2]);
    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_38(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1048576; oi++) {
    const d0: i32 = (oi / 1048576) % 1;
    const d1: i32 = (oi / 262144) % 4;
    const d2: i32 = (oi / 2048) % 128;
    const d3: i32 = (oi / 16) % 128;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + (d1 * 16) + d4]);
    _out[oi] = t0;
  }
}

function kernel_39(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let d2: i32 = 0; d2 < 128; d2++) {
        for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
          const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
          for (let mi: i32 = 0; mi < m_end; mi++) {
            const d3: i32 = m_blk * 1 + mi;
            const _row_base: i32 = d0 * 8192 + d1 * 2048 + d2 * 16 + d3 * 16;
            for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
              const n_tile: i32 = 16 - n_blk * 16 < 16 ? 16 - n_blk * 16 : 16;
              for (let ni_grp: i32 = 0; ni_grp < (n_tile / 16); ni_grp++) {
                const ni_base: i32 = n_blk * 16 + ni_grp * 16;
                let _acc0: f32 = f32(0.0);
                let _acc1: f32 = f32(0.0);
                let _acc2: f32 = f32(0.0);
                let _acc3: f32 = f32(0.0);
                let _acc4: f32 = f32(0.0);
                let _acc5: f32 = f32(0.0);
                let _acc6: f32 = f32(0.0);
                let _acc7: f32 = f32(0.0);
                let _acc8: f32 = f32(0.0);
                let _acc9: f32 = f32(0.0);
                let _acc10: f32 = f32(0.0);
                let _acc11: f32 = f32(0.0);
                let _acc12: f32 = f32(0.0);
                let _acc13: f32 = f32(0.0);
                let _acc14: f32 = f32(0.0);
                let _acc15: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    {
                      const d4: i32 = ni_base + 0;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc0 = _acc0 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 1;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc1 = _acc1 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 2;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc2 = _acc2 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 3;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc3 = _acc3 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 4;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc4 = _acc4 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 5;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc5 = _acc5 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 6;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc6 = _acc6 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 7;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc7 = _acc7 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 8;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc8 = _acc8 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 9;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc9 = _acc9 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 10;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc10 = _acc10 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 11;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc11 = _acc11 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 12;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc12 = _acc12 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 13;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc13 = _acc13 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 14;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc14 = _acc14 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 15;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc15 = _acc15 + t2;
                    }
                  }
                }
                unchecked(_out[_row_base + ni_base + 0] = _acc0);
                unchecked(_out[_row_base + ni_base + 1] = _acc1);
                unchecked(_out[_row_base + ni_base + 2] = _acc2);
                unchecked(_out[_row_base + ni_base + 3] = _acc3);
                unchecked(_out[_row_base + ni_base + 4] = _acc4);
                unchecked(_out[_row_base + ni_base + 5] = _acc5);
                unchecked(_out[_row_base + ni_base + 6] = _acc6);
                unchecked(_out[_row_base + ni_base + 7] = _acc7);
                unchecked(_out[_row_base + ni_base + 8] = _acc8);
                unchecked(_out[_row_base + ni_base + 9] = _acc9);
                unchecked(_out[_row_base + ni_base + 10] = _acc10);
                unchecked(_out[_row_base + ni_base + 11] = _acc11);
                unchecked(_out[_row_base + ni_base + 12] = _acc12);
                unchecked(_out[_row_base + ni_base + 13] = _acc13);
                unchecked(_out[_row_base + ni_base + 14] = _acc14);
                unchecked(_out[_row_base + ni_base + 15] = _acc15);
              }
              for (let ni: i32 = (n_tile / 16) * 16; ni < n_tile; ni++) {
                const d4: i32 = n_blk * 16 + ni;
                let _acc_r: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d5 * 16) + d4]);
                    const t2: f32 = (t0 * t1);
                    _acc_r = _acc_r + t2;
                  }
                }
                unchecked(_out[_row_base + n_blk * 16 + ni] = _acc_r);
              }
            }
          }
        }
      }
    }
  }
}

function kernel_40(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = (oi / 16) % 4;
    const d3: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d2 * 2048) + (d1 * 16) + d3]);
    _out[oi] = t0;
  }
}

function kernel_41(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d2]);
    _out[oi] = t0;
  }
}

function kernel_42(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 524288; oi++) {
    const d0: i32 = (oi / 524288) % 1;
    const d1: i32 = (oi / 4096) % 128;
    const d2: i32 = (oi / 64) % 64;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d2]);
    _out[oi] = t0;
  }
}

function kernel_43(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 524288; oi++) {
    const d0: i32 = (oi / 524288) % 1;
    const d1: i32 = (oi / 4096) % 128;
    const d2: i32 = (oi / 64) % 64;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d2 * 64) + d3]);
    _out[oi] = t0;
  }
}

function kernel_44(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 8192 + d1 * 64 + d2 * 64;
          for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
            const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d4 * 64) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_45(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 8192) + (d1 * 64) + d2]);
    const t2: f32 = unchecked(_in2[d2]);
    const t3: f32 = (t1 + t2);
    const t4: f32 = (t0 + t3);
    _out[oi] = t4;
  }
}

function kernel_46(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 2097152; oi++) {
    const d0: i32 = (oi / 2097152) % 1;
    const d1: i32 = (oi / 16384) % 128;
    const d2: i32 = (oi / 256) % 64;
    const d3: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d2]);
    const t1: f32 = unchecked(_in1[d2]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[d2]);
    const t4: f32 = (t2 + t3);
    _out[oi] = t4;
  }
}

function kernel_47(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 2097152; oi++) {
    const d0: i32 = (oi / 2097152) % 1;
    const d1: i32 = (oi / 16384) % 128;
    const d2: i32 = (oi / 256) % 64;
    const d3: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[(d2 * 256) + d3]);
    _out[oi] = t0;
  }
}

function kernel_48(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 32768 + d1 * 256 + d2 * 256;
          for (let n_blk: i32 = 0; n_blk < 8; n_blk++) {
            const n_tile: i32 = 256 - n_blk * 32 < 32 ? 256 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 256) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_49(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[(d0 * 32768) + (d1 * 256) + d2]);
    const t1: f32 = unchecked(_in1[d2]);
    const t2: f32 = (t0 + t1);
    _out[oi] = t2;
  }
}

function kernel_50(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in0[oi]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_51(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_52(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = unchecked(_in2[oi]);
    const t3: f32 = (t1 * t2);
    const t4: f32 = (t0 + t3);
    _out[oi] = t4;
  }
}

function kernel_53(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = (-t2);
    _out[oi] = t3;
  }
}

function kernel_54(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 > t1 ? t0 : t1);
    const t3: f32 = (-t2);
    _out[oi] = t3;
  }
}

function kernel_55(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 > t1 ? t0 : t1);
    _out[oi] = t2;
  }
}

function kernel_56(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_57(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = f32(Math.pow(2.0, f64(t0)));
    _out[oi] = t1;
  }
}

function kernel_58(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (-t1);
    const t3: f32 = (t0 + t2);
    _out[oi] = t3;
  }
}

function kernel_59(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 + t1);
    _out[oi] = t2;
  }
}

function kernel_60(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = (f32(1.0) / t0);
    _out[oi] = t1;
  }
}

function kernel_61(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_62(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = unchecked(_in2[oi]);
    const t3: f32 = (t1 * t2);
    const t4: f32 = (t0 + t3);
    _out[oi] = t4;
  }
}

function kernel_63(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 2097152; oi++) {
    const d0: i32 = (oi / 2097152) % 1;
    const d1: i32 = (oi / 16384) % 128;
    const d2: i32 = (oi / 64) % 256;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 32768) + (d1 * 256) + d2]);
    const t1: f32 = unchecked(_in1[(d0 * 32768) + (d1 * 256) + d2]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_64(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 2097152; oi++) {
    const d0: i32 = (oi / 2097152) % 1;
    const d1: i32 = (oi / 16384) % 128;
    const d2: i32 = (oi / 64) % 256;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d2 * 64) + d3]);
    _out[oi] = t0;
  }
}

function kernel_65(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 8192 + d1 * 64 + d2 * 64;
          for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
            const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 8; k_blk++) {
                const k_end: i32 = 256 - k_blk * 32 < 32 ? 256 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 8; k_blk++) {
                const k_end: i32 = 256 - k_blk * 32 < 32 ? 256 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d4 * 64) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_66(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 811008; oi++) {
    const d0: i32 = (oi / 811008) % 1;
    const d1: i32 = (oi / 6336) % 128;
    const d2: i32 = (oi / 99) % 64;
    const d3: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d2]);
    const t1: f32 = unchecked(_in1[d2]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[d2]);
    const t4: f32 = (t2 + t3);
    _out[oi] = t4;
  }
}

function kernel_67(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 811008; oi++) {
    const d0: i32 = (oi / 811008) % 1;
    const d1: i32 = (oi / 6336) % 128;
    const d2: i32 = (oi / 99) % 64;
    const d3: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[(d3 * 64) + d2]);
    _out[oi] = t0;
  }
}

function kernel_68(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 12672 + d1 * 99 + d2 * 99;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 99 - n_blk * 32 < 32 ? 99 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d4 * 99) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_69(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d2]);
    _out[oi] = t0;
  }
}

function kernel_70(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_71(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(-Infinity);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = t0 > _acc0 ? t0 : _acc0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(-Infinity);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                _acc_r = t0 > _acc_r ? t0 : _acc_r;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_72(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 128) + d1]);
    const t2: f32 = (-t1);
    const t3: f32 = (t0 + t2);
    _out[oi] = t3;
  }
}

function kernel_73(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_74(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                const t1: f32 = f32(Math.pow(2.0, f64(t0)));
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t1;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                const t1: f32 = f32(Math.pow(2.0, f64(t0)));
                _acc_r = _acc_r + t1;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_75(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 128; oi++) {
    const d0: i32 = (oi / 128) % 1;
    const d1: i32 = (oi / 1) % 128;
    const d2: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = f32(Math.log2(f64(t0)));
    _out[oi] = t1;
  }
}

function kernel_76(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 128) + d1]);
    const t2: f32 = unchecked(_in2[0]);
    const t3: f32 = (t1 * t2);
    const t4: f32 = (-t3);
    const t5: f32 = (t0 + t4);
    _out[oi] = t5;
  }
}

function kernel_77(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                const t1: f32 = unchecked(_in1[(d0 * 12672) + (d1 * 99) + d3]);
                const t2: f32 = (t0 * t1);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t2;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                const t1: f32 = unchecked(_in1[(d0 * 12672) + (d1 * 99) + d3]);
                const t2: f32 = (t0 * t1);
                _acc_r = _acc_r + t2;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_78(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d3 * 128) + d1 + d2]);
                  _acc0 = _acc0 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                const t0: f32 = unchecked(_in0[(d3 * 128) + d1 + d2]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_79(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1; oi++) {
    const t0: f32 = unchecked(_in0[0]);
    _out[oi] = t0;
  }
}

function kernel_80(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 1; oi++) {
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_81(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 1; oi++) {
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = (-t2);
    _out[oi] = t3;
  }
}

function kernel_82(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 128; oi++) {
    const d0: i32 = (oi / 128) % 1;
    const d1: i32 = (oi / 1) % 128;
    const d2: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[0]);
    _out[oi] = t0;
  }
}

function kernel_83(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_84(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 128; oi++) {
    const d0: i32 = (oi / 128) % 1;
    const d1: i32 = (oi / 1) % 128;
    const d2: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = (-t0);
    _out[oi] = t1;
  }
}

function kernel_85(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d3 * 128) + d1 + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 128) + d1 + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc0 = _acc0 + t2;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                const t0: f32 = unchecked(_in0[(d3 * 128) + d1 + d2]);
                const t1: f32 = unchecked(_in1[(d3 * 128) + d1 + d2]);
                const t2: f32 = (t0 * t1);
                _acc_r = _acc_r + t2;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_86(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 1 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d0 * 128) + d3 + d2]);
                  _acc0 = _acc0 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 128) + d3 + d2]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_87(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1; oi++) {
    let acc: f32 = f32(0.0);
    const d0: i32 = (oi / 1) % 1;
    const d1: i32 = (oi / 1) % 1;
    const d2: i32 = oi % 1;
    for (let d3: i32 = 0; d3 < 1; d3++) {
      const t0: f32 = unchecked(_in0[d0 + d1 + d3]);
      acc = acc + t0;
    }
    _out[oi] = acc;
  }
}

function kernel_88(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array, _in4: Float32Array, _in5: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[(d0 * 128) + d1]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[(d0 * 128) + d1]);
    const t4: f32 = unchecked(_in3[0]);
    const t5: f32 = (t3 * t4);
    const t6: f32 = (f32(1.0) / t5);
    const t7: f32 = (t2 * t6);
    const t8: f32 = unchecked(_in4[oi]);
    const t9: f32 = f32(Math.pow(2.0, f64(t8)));
    const t10: f32 = unchecked(_in5[0]);
    const t11: f32 = (t9 * t10);
    const t12: f32 = (t7 * t11);
    _out[oi] = t12;
  }
}

function kernel_89(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 12672 + d1 * 99;
        for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
          const n_tile: i32 = 99 - n_blk * 32 < 32 ? 99 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc0 = _acc0 + t2;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc1 = _acc1 + t2;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc2 = _acc2 + t2;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc3 = _acc3 + t2;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc4 = _acc4 + t2;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc5 = _acc5 + t2;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc6 = _acc6 + t2;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc7 = _acc7 + t2;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc8 = _acc8 + t2;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc9 = _acc9 + t2;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc10 = _acc10 + t2;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc11 = _acc11 + t2;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc12 = _acc12 + t2;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc13 = _acc13 + t2;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc14 = _acc14 + t2;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc15 = _acc15 + t2;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc16 = _acc16 + t2;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc17 = _acc17 + t2;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc18 = _acc18 + t2;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc19 = _acc19 + t2;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc20 = _acc20 + t2;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc21 = _acc21 + t2;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc22 = _acc22 + t2;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc23 = _acc23 + t2;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc24 = _acc24 + t2;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc25 = _acc25 + t2;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc26 = _acc26 + t2;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc27 = _acc27 + t2;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc28 = _acc28 + t2;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc29 = _acc29 + t2;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc30 = _acc30 + t2;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc31 = _acc31 + t2;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                const t0: f32 = unchecked(_in0[(d3 * 12672) + (d1 * 99) + d2]);
                const t1: f32 = unchecked(_in1[(d3 * 12672) + (d1 * 99) + d2]);
                const t2: f32 = (t0 * t1);
                _acc_r = _acc_r + t2;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_90(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 99 + d1 * 99;
        for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
          const n_tile: i32 = 99 - n_blk * 32 < 32 ? 99 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc0 = _acc0 + t0;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc1 = _acc1 + t0;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc2 = _acc2 + t0;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc3 = _acc3 + t0;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc4 = _acc4 + t0;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc5 = _acc5 + t0;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc6 = _acc6 + t0;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc7 = _acc7 + t0;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc8 = _acc8 + t0;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc9 = _acc9 + t0;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc10 = _acc10 + t0;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc11 = _acc11 + t0;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc12 = _acc12 + t0;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc13 = _acc13 + t0;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc14 = _acc14 + t0;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc15 = _acc15 + t0;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc16 = _acc16 + t0;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc17 = _acc17 + t0;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc18 = _acc18 + t0;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc19 = _acc19 + t0;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc20 = _acc20 + t0;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc21 = _acc21 + t0;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc22 = _acc22 + t0;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc23 = _acc23 + t0;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc24 = _acc24 + t0;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc25 = _acc25 + t0;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc26 = _acc26 + t0;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc27 = _acc27 + t0;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc28 = _acc28 + t0;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc29 = _acc29 + t0;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc30 = _acc30 + t0;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                  _acc31 = _acc31 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 12672) + (d3 * 99) + d2]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_91(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 1 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 99) + (d1 * 99) + d3]);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 99) + (d1 * 99) + d3]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_92(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = unchecked(_in2[0]);
    const t3: f32 = (t1 * t2);
    const t4: f32 = (t0 + t3);
    _out[oi] = t4;
  }
}

function kernel_93(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 12672; oi++) {
    const d0: i32 = (oi / 12672) % 1;
    const d1: i32 = (oi / 99) % 128;
    const d2: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = unchecked(_in2[(d0 * 128) + d1]);
    const t3: f32 = (t1 < t2 ? f32(1.0) : f32(0.0));
    const t4: f32 = (-t3);
    const t5: f32 = (t0 + t4);
    _out[oi] = t5;
  }
}

function kernel_94(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array): void {
  for (let oi: i32 = 0; oi < 811008; oi++) {
    const d0: i32 = (oi / 811008) % 1;
    const d1: i32 = (oi / 6336) % 128;
    const d2: i32 = (oi / 99) % 64;
    const d3: i32 = oi % 99;
    const t0: f32 = unchecked(_in0[(d0 * 12672) + (d1 * 99) + d3]);
    const t1: f32 = unchecked(_in1[(d0 * 128) + d1]);
    const t2: f32 = (-t1);
    const t3: f32 = unchecked(_in2[(d0 * 12672) + (d1 * 99) + d3]);
    const t4: f32 = unchecked(_in3[(d0 * 128) + d1]);
    const t5: f32 = (f32(1.0) / t4);
    const t6: f32 = (t3 * t5);
    const t7: f32 = (t2 * t6);
    const t8: f32 = (t0 + t7);
    _out[oi] = t8;
  }
}

function kernel_95(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 6336 + d1 * 6336 + d2 * 99;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 99 - n_blk * 32 < 32 ? 99 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 99) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_96(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 8192 + d1 * 64 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d2 * 99) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d2 * 99) + d4]);
                  const t2: f32 = (t0 * t1);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 99 - k_blk * 32 < 32 ? 99 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d2 * 99) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d2 * 99) + d4]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_97(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 64 + d1 * 64;
        for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
          const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc0 = _acc0 + t0;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc1 = _acc1 + t0;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc2 = _acc2 + t0;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc3 = _acc3 + t0;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc4 = _acc4 + t0;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc5 = _acc5 + t0;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc6 = _acc6 + t0;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc7 = _acc7 + t0;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc8 = _acc8 + t0;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc9 = _acc9 + t0;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc10 = _acc10 + t0;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc11 = _acc11 + t0;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc12 = _acc12 + t0;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc13 = _acc13 + t0;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc14 = _acc14 + t0;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc15 = _acc15 + t0;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc16 = _acc16 + t0;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc17 = _acc17 + t0;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc18 = _acc18 + t0;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc19 = _acc19 + t0;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc20 = _acc20 + t0;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc21 = _acc21 + t0;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc22 = _acc22 + t0;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc23 = _acc23 + t0;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc24 = _acc24 + t0;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc25 = _acc25 + t0;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc26 = _acc26 + t0;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc27 = _acc27 + t0;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc28 = _acc28 + t0;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc29 = _acc29 + t0;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc30 = _acc30 + t0;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  _acc31 = _acc31 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_98(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 64; oi++) {
    const d0: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[d0]);
    _out[oi] = t0;
  }
}

function kernel_99(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[d2]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_100(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 64 + d1 * 64;
        for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
          const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc0 = _acc0 + t2;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc1 = _acc1 + t2;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc2 = _acc2 + t2;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc3 = _acc3 + t2;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc4 = _acc4 + t2;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc5 = _acc5 + t2;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc6 = _acc6 + t2;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc7 = _acc7 + t2;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc8 = _acc8 + t2;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc9 = _acc9 + t2;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc10 = _acc10 + t2;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc11 = _acc11 + t2;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc12 = _acc12 + t2;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc13 = _acc13 + t2;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc14 = _acc14 + t2;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc15 = _acc15 + t2;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc16 = _acc16 + t2;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc17 = _acc17 + t2;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc18 = _acc18 + t2;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc19 = _acc19 + t2;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc20 = _acc20 + t2;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc21 = _acc21 + t2;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc22 = _acc22 + t2;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc23 = _acc23 + t2;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc24 = _acc24 + t2;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc25 = _acc25 + t2;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc26 = _acc26 + t2;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc27 = _acc27 + t2;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc28 = _acc28 + t2;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc29 = _acc29 + t2;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc30 = _acc30 + t2;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                  const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc31 = _acc31 + t2;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 64) + d2]);
                const t1: f32 = unchecked(_in1[(d0 * 8192) + (d3 * 64) + d2]);
                const t2: f32 = (t0 * t1);
                _acc_r = _acc_r + t2;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_101(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 128 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
              const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                const t1: f32 = unchecked(_in1[(d0 * 8192) + (d1 * 64) + d3]);
                const t2: f32 = (t0 * t1);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t2;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
              const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
                const t1: f32 = unchecked(_in1[(d0 * 8192) + (d1 * 64) + d3]);
                const t2: f32 = (t0 * t1);
                _acc_r = _acc_r + t2;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_102(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array): void {
  for (let oi: i32 = 0; oi < 128; oi++) {
    const d0: i32 = (oi / 128) % 1;
    const d1: i32 = (oi / 1) % 128;
    const d2: i32 = oi % 1;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = unchecked(_in1[oi]);
    const t3: f32 = (t1 * t2);
    const t4: f32 = (-t3);
    const t5: f32 = (t0 * t4);
    const t6: f32 = unchecked(_in2[0]);
    const t7: f32 = unchecked(_in3[oi]);
    const t8: f32 = f32(Math.sqrt(f64(t7)));
    const t9: f32 = (t6 * t8);
    const t10: f32 = (f32(1.0) / t9);
    const t11: f32 = (t5 * t10);
    _out[oi] = t11;
  }
}

function kernel_103(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 128) + d1]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_104(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 128) + d1]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[oi]);
    const t4: f32 = unchecked(_in3[oi]);
    const t5: f32 = (t3 * t4);
    const t6: f32 = (t2 + t5);
    const t7: f32 = unchecked(_in2[oi]);
    const t8: f32 = unchecked(_in3[oi]);
    const t9: f32 = (t7 * t8);
    const t10: f32 = (t6 + t9);
    _out[oi] = t10;
  }
}

function kernel_105(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 128) + d1]);
    const t2: f32 = unchecked(_in2[0]);
    const t3: f32 = (t1 * t2);
    const t4: f32 = (t0 + t3);
    _out[oi] = t4;
  }
}

function kernel_106(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 2097152; oi++) {
    const d0: i32 = (oi / 2097152) % 1;
    const d1: i32 = (oi / 16384) % 128;
    const d2: i32 = (oi / 64) % 256;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
    _out[oi] = t0;
  }
}

function kernel_107(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 32; m_blk++) {
        const m_end: i32 = 256 - m_blk * 8 < 8 ? 256 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 16384 + d1 * 16384 + d2 * 64;
          for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
            const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 64) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_108(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 32; m_blk++) {
        const m_end: i32 = 256 - m_blk * 8 < 8 ? 256 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 32768 + d1 * 256 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d2 * 64) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d2 * 64) + d4]);
                  const t2: f32 = (t0 * t1);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d2 * 64) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d2 * 64) + d4]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_109(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 16384; oi++) {
    const d0: i32 = (oi / 64) % 256;
    const d1: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 64) + d1]);
    _out[oi] = t0;
  }
}

function kernel_110(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[(d0 * 32768) + (d1 * 256) + d2]);
    _out[oi] = t0;
  }
}

function kernel_111(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 32768 + d1 * 256;
        for (let n_blk: i32 = 0; n_blk < 8; n_blk++) {
          const n_tile: i32 = 256 - n_blk * 32 < 32 ? 256 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc0 = _acc0 + t0;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc1 = _acc1 + t0;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc2 = _acc2 + t0;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc3 = _acc3 + t0;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc4 = _acc4 + t0;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc5 = _acc5 + t0;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc6 = _acc6 + t0;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc7 = _acc7 + t0;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc8 = _acc8 + t0;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc9 = _acc9 + t0;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc10 = _acc10 + t0;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc11 = _acc11 + t0;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc12 = _acc12 + t0;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc13 = _acc13 + t0;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc14 = _acc14 + t0;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc15 = _acc15 + t0;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc16 = _acc16 + t0;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc17 = _acc17 + t0;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc18 = _acc18 + t0;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc19 = _acc19 + t0;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc20 = _acc20 + t0;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc21 = _acc21 + t0;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc22 = _acc22 + t0;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc23 = _acc23 + t0;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc24 = _acc24 + t0;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc25 = _acc25 + t0;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc26 = _acc26 + t0;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc27 = _acc27 + t0;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc28 = _acc28 + t0;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc29 = _acc29 + t0;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc30 = _acc30 + t0;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  _acc31 = _acc31 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_112(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 256 + d1 * 256;
        for (let n_blk: i32 = 0; n_blk < 8; n_blk++) {
          const n_tile: i32 = 256 - n_blk * 32 < 32 ? 256 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc0 = _acc0 + t0;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc1 = _acc1 + t0;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc2 = _acc2 + t0;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc3 = _acc3 + t0;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc4 = _acc4 + t0;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc5 = _acc5 + t0;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc6 = _acc6 + t0;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc7 = _acc7 + t0;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc8 = _acc8 + t0;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc9 = _acc9 + t0;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc10 = _acc10 + t0;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc11 = _acc11 + t0;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc12 = _acc12 + t0;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc13 = _acc13 + t0;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc14 = _acc14 + t0;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc15 = _acc15 + t0;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc16 = _acc16 + t0;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc17 = _acc17 + t0;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc18 = _acc18 + t0;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc19 = _acc19 + t0;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc20 = _acc20 + t0;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc21 = _acc21 + t0;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc22 = _acc22 + t0;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc23 = _acc23 + t0;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc24 = _acc24 + t0;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc25 = _acc25 + t0;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc26 = _acc26 + t0;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc27 = _acc27 + t0;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc28 = _acc28 + t0;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc29 = _acc29 + t0;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc30 = _acc30 + t0;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                  _acc31 = _acc31 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 32768) + (d3 * 256) + d2]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_113(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 1 + d1 * 1;
        for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
          const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
            const ni_base: i32 = n_blk * 1 + ni_grp * 1;
            let _acc0: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 8; k_blk++) {
              const k_end: i32 = 256 - k_blk * 32 < 32 ? 256 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 256) + (d1 * 256) + d3]);
                {
                  const d2: i32 = ni_base + 0;
                  _acc0 = _acc0 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
          }
          for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 1 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 8; k_blk++) {
              const k_end: i32 = 256 - k_blk * 32 < 32 ? 256 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 256) + (d1 * 256) + d3]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_114(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
      const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 8 + mi;
        const _row_base: i32 = d0 * 32768 + d1 * 256;
        for (let n_blk: i32 = 0; n_blk < 8; n_blk++) {
          const n_tile: i32 = 256 - n_blk * 32 < 32 ? 256 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc0 = _acc0 + t2;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc1 = _acc1 + t2;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc2 = _acc2 + t2;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc3 = _acc3 + t2;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc4 = _acc4 + t2;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc5 = _acc5 + t2;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc6 = _acc6 + t2;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc7 = _acc7 + t2;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc8 = _acc8 + t2;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc9 = _acc9 + t2;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc10 = _acc10 + t2;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc11 = _acc11 + t2;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc12 = _acc12 + t2;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc13 = _acc13 + t2;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc14 = _acc14 + t2;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc15 = _acc15 + t2;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc16 = _acc16 + t2;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc17 = _acc17 + t2;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc18 = _acc18 + t2;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc19 = _acc19 + t2;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc20 = _acc20 + t2;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc21 = _acc21 + t2;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc22 = _acc22 + t2;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc23 = _acc23 + t2;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc24 = _acc24 + t2;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc25 = _acc25 + t2;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc26 = _acc26 + t2;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc27 = _acc27 + t2;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc28 = _acc28 + t2;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc29 = _acc29 + t2;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc30 = _acc30 + t2;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                  const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                  const t2: f32 = (t0 * t1);
                  _acc31 = _acc31 + t2;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
              const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 1 + ki;
                const t0: f32 = unchecked(_in0[(d3 * 32768) + (d1 * 256) + d2]);
                const t1: f32 = unchecked(_in1[(d3 * 32768) + (d1 * 256) + d2]);
                const t2: f32 = (t0 * t1);
                _acc_r = _acc_r + t2;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_115(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[oi]);
    const t4: f32 = unchecked(_in2[oi]);
    const t5: f32 = (t3 * t4);
    const t6: f32 = (-t5);
    const t7: f32 = (t2 * t6);
    _out[oi] = t7;
  }
}

function kernel_116(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 + t1);
    const t3: f32 = unchecked(_in2[oi]);
    const t4: f32 = f32(Math.pow(2.0, f64(t3)));
    const t5: f32 = unchecked(_in3[0]);
    const t6: f32 = (t4 * t5);
    const t7: f32 = (t2 * t6);
    _out[oi] = t7;
  }
}

function kernel_117(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 < t1 ? f32(1.0) : f32(0.0));
    _out[oi] = t2;
  }
}

function kernel_118(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = unchecked(_in2[oi]);
    const t3: f32 = (-t2);
    const t4: f32 = (t1 + t3);
    const t5: f32 = (t0 * t4);
    const t6: f32 = (-t5);
    _out[oi] = t6;
  }
}

function kernel_119(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array, _in4: Float32Array, _in5: Float32Array, _in6: Float32Array): void {
  for (let oi: i32 = 0; oi < 32768; oi++) {
    const d0: i32 = (oi / 32768) % 1;
    const d1: i32 = (oi / 256) % 128;
    const d2: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[oi]);
    const t4: f32 = (t2 + t3);
    const t5: f32 = unchecked(_in3[oi]);
    const t6: f32 = unchecked(_in4[oi]);
    const t7: f32 = (t5 * t6);
    const t8: f32 = (t4 + t7);
    const t9: f32 = unchecked(_in5[oi]);
    const t10: f32 = unchecked(_in6[oi]);
    const t11: f32 = (t9 * t10);
    const t12: f32 = (t8 + t11);
    const t13: f32 = unchecked(_in5[oi]);
    const t14: f32 = unchecked(_in6[oi]);
    const t15: f32 = (t13 * t14);
    const t16: f32 = (t12 + t15);
    _out[oi] = t16;
  }
}

function kernel_120(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 256; oi++) {
    const d0: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[d0]);
    _out[oi] = t0;
  }
}

function kernel_121(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 2097152; oi++) {
    const d0: i32 = (oi / 2097152) % 1;
    const d1: i32 = (oi / 16384) % 128;
    const d2: i32 = (oi / 256) % 64;
    const d3: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[(d0 * 32768) + (d1 * 256) + d3]);
    _out[oi] = t0;
  }
}

function kernel_122(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 16384 + d1 * 16384 + d2 * 256;
          for (let n_blk: i32 = 0; n_blk < 8; n_blk++) {
            const n_tile: i32 = 256 - n_blk * 32 < 32 ? 256 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d4 * 16384) + (d2 * 256) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_123(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 8192 + d1 * 64 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 8; k_blk++) {
                const k_end: i32 = 256 - k_blk * 32 < 32 ? 256 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d2 * 256) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d2 * 256) + d4]);
                  const t2: f32 = (t0 * t1);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 8; k_blk++) {
                const k_end: i32 = 256 - k_blk * 32 < 32 ? 256 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 2097152) + (d1 * 16384) + (d2 * 256) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 2097152) + (d1 * 16384) + (d2 * 256) + d4]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_124(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 16384; oi++) {
    const d0: i32 = (oi / 256) % 64;
    const d1: i32 = oi % 256;
    const t0: f32 = unchecked(_in0[(d0 * 256) + d1]);
    _out[oi] = t0;
  }
}

function kernel_125(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 + t1);
    const t3: f32 = unchecked(_in2[(d0 * 128) + d1]);
    const t4: f32 = unchecked(_in3[0]);
    const t5: f32 = (t3 * t4);
    const t6: f32 = (t2 + t5);
    _out[oi] = t6;
  }
}

function kernel_126(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 524288; oi++) {
    const d0: i32 = (oi / 524288) % 1;
    const d1: i32 = (oi / 4096) % 128;
    const d2: i32 = (oi / 64) % 64;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
    _out[oi] = t0;
  }
}

function kernel_127(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 4096 + d1 * 4096 + d2 * 64;
          for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
            const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 524288) + (d4 * 4096) + (d2 * 64) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_128(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 8192 + d1 * 64 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d2 * 64) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d2 * 64) + d4]);
                  const t2: f32 = (t0 * t1);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 524288) + (d1 * 4096) + (d2 * 64) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 524288) + (d1 * 4096) + (d2 * 64) + d4]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_129(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 4096; oi++) {
    const d0: i32 = (oi / 64) % 64;
    const d1: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 64) + d1]);
    _out[oi] = t0;
  }
}

function kernel_130(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = (oi / 16) % 4;
    const d3: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + (d2 * 16) + d3]);
    _out[oi] = t0;
  }
}

function kernel_131(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1048576; oi++) {
    const d0: i32 = (oi / 1048576) % 1;
    const d1: i32 = (oi / 262144) % 4;
    const d2: i32 = (oi / 2048) % 128;
    const d3: i32 = (oi / 16) % 128;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d2 * 64) + (d1 * 16) + d4]);
    _out[oi] = t0;
  }
}

function kernel_132(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let d2: i32 = 0; d2 < 1; d2++) {
        for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
          const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
          for (let mi: i32 = 0; mi < m_end; mi++) {
            const d3: i32 = m_blk * 8 + mi;
            const _row_base: i32 = d0 * 8192 + d1 * 2048 + d2 * 2048 + d3 * 16;
            for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
              const n_tile: i32 = 16 - n_blk * 16 < 16 ? 16 - n_blk * 16 : 16;
              for (let ni_grp: i32 = 0; ni_grp < (n_tile / 16); ni_grp++) {
                const ni_base: i32 = n_blk * 16 + ni_grp * 16;
                let _acc0: f32 = f32(0.0);
                let _acc1: f32 = f32(0.0);
                let _acc2: f32 = f32(0.0);
                let _acc3: f32 = f32(0.0);
                let _acc4: f32 = f32(0.0);
                let _acc5: f32 = f32(0.0);
                let _acc6: f32 = f32(0.0);
                let _acc7: f32 = f32(0.0);
                let _acc8: f32 = f32(0.0);
                let _acc9: f32 = f32(0.0);
                let _acc10: f32 = f32(0.0);
                let _acc11: f32 = f32(0.0);
                let _acc12: f32 = f32(0.0);
                let _acc13: f32 = f32(0.0);
                let _acc14: f32 = f32(0.0);
                let _acc15: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    {
                      const d4: i32 = ni_base + 0;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc0 = _acc0 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 1;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc1 = _acc1 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 2;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc2 = _acc2 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 3;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc3 = _acc3 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 4;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc4 = _acc4 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 5;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc5 = _acc5 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 6;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc6 = _acc6 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 7;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc7 = _acc7 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 8;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc8 = _acc8 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 9;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc9 = _acc9 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 10;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc10 = _acc10 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 11;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc11 = _acc11 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 12;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc12 = _acc12 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 13;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc13 = _acc13 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 14;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc14 = _acc14 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 15;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc15 = _acc15 + t2;
                    }
                  }
                }
                unchecked(_out[_row_base + ni_base + 0] = _acc0);
                unchecked(_out[_row_base + ni_base + 1] = _acc1);
                unchecked(_out[_row_base + ni_base + 2] = _acc2);
                unchecked(_out[_row_base + ni_base + 3] = _acc3);
                unchecked(_out[_row_base + ni_base + 4] = _acc4);
                unchecked(_out[_row_base + ni_base + 5] = _acc5);
                unchecked(_out[_row_base + ni_base + 6] = _acc6);
                unchecked(_out[_row_base + ni_base + 7] = _acc7);
                unchecked(_out[_row_base + ni_base + 8] = _acc8);
                unchecked(_out[_row_base + ni_base + 9] = _acc9);
                unchecked(_out[_row_base + ni_base + 10] = _acc10);
                unchecked(_out[_row_base + ni_base + 11] = _acc11);
                unchecked(_out[_row_base + ni_base + 12] = _acc12);
                unchecked(_out[_row_base + ni_base + 13] = _acc13);
                unchecked(_out[_row_base + ni_base + 14] = _acc14);
                unchecked(_out[_row_base + ni_base + 15] = _acc15);
              }
              for (let ni: i32 = (n_tile / 16) * 16; ni < n_tile; ni++) {
                const d4: i32 = n_blk * 16 + ni;
                let _acc_r: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 16) + d4]);
                    const t2: f32 = (t0 * t1);
                    _acc_r = _acc_r + t2;
                  }
                }
                unchecked(_out[_row_base + n_blk * 16 + ni] = _acc_r);
              }
            }
          }
        }
      }
    }
  }
}

function kernel_133(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let d2: i32 = 0; d2 < 128; d2++) {
        for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
          const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
          for (let mi: i32 = 0; mi < m_end; mi++) {
            const d3: i32 = m_blk * 8 + mi;
            const _row_base: i32 = d0 * 65536 + d1 * 16384 + d2 * 128 + d3 * 1;
            for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
              const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
              for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
                const ni_base: i32 = n_blk * 1 + ni_grp * 1;
                let _acc0: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                  const k_end: i32 = 16 - k_blk * 16 < 16 ? 16 - k_blk * 16 : 16;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 16 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 16) + d5]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 16) + d5]);
                    const t2: f32 = (t0 * t1);
                    {
                      const d4: i32 = ni_base + 0;
                      _acc0 = _acc0 + t2;
                    }
                  }
                }
                unchecked(_out[_row_base + ni_base + 0] = _acc0);
              }
              for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
                const d4: i32 = n_blk * 1 + ni;
                let _acc_r: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                  const k_end: i32 = 16 - k_blk * 16 < 16 ? 16 - k_blk * 16 : 16;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 16 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 16) + d5]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 16) + d5]);
                    const t2: f32 = (t0 * t1);
                    _acc_r = _acc_r + t2;
                  }
                }
                unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
              }
            }
          }
        }
      }
    }
  }
}

function kernel_134(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 512 + d1 * 128 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  const t2: f32 = (t0 * t1);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_135(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array, _in4: Float32Array, _in5: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 512) + (d1 * 128) + d2]);
    const t2: f32 = (t0 * t1);
    const t3: f32 = unchecked(_in2[(d0 * 512) + (d1 * 128) + d2]);
    const t4: f32 = unchecked(_in3[(d0 * 512) + (d1 * 128) + d2]);
    const t5: f32 = unchecked(_in3[(d0 * 512) + (d1 * 128) + d2]);
    const t6: f32 = (t4 * t5);
    const t7: f32 = (-t6);
    const t8: f32 = (t3 * t7);
    const t9: f32 = (t2 + t8);
    const t10: f32 = unchecked(_in4[oi]);
    const t11: f32 = f32(Math.pow(2.0, f64(t10)));
    const t12: f32 = unchecked(_in5[0]);
    const t13: f32 = (t11 * t12);
    const t14: f32 = (t9 * t13);
    _out[oi] = t14;
  }
}

function kernel_136(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 65536 + d1 * 16384 + d2 * 128;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 1 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 1 + ki;
                  const t0: f32 = unchecked(_in0[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                  const t1: f32 = unchecked(_in1[(d4 * 65536) + (d1 * 16384) + (d2 * 128) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_137(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 16384 + d1 * 16384 + d2 * 128;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 4 - k_blk * 4 < 4 ? 4 - k_blk * 4 : 4;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 4 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc0 = _acc0 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc1 = _acc1 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc2 = _acc2 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc3 = _acc3 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc4 = _acc4 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc5 = _acc5 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc6 = _acc6 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc7 = _acc7 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc8 = _acc8 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc9 = _acc9 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc10 = _acc10 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc11 = _acc11 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc12 = _acc12 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc13 = _acc13 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc14 = _acc14 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc15 = _acc15 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc16 = _acc16 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc17 = _acc17 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc18 = _acc18 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc19 = _acc19 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc20 = _acc20 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc21 = _acc21 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc22 = _acc22 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc23 = _acc23 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc24 = _acc24 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc25 = _acc25 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc26 = _acc26 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc27 = _acc27 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc28 = _acc28 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc29 = _acc29 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc30 = _acc30 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc31 = _acc31 + t0;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 4 - k_blk * 4 < 4 ? 4 - k_blk * 4 : 4;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 4 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                  _acc_r = _acc_r + t0;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_138(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 128 + d1 * 128 + d2 * 128;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc0 = _acc0 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc1 = _acc1 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc2 = _acc2 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc3 = _acc3 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc4 = _acc4 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc5 = _acc5 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc6 = _acc6 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc7 = _acc7 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc8 = _acc8 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc9 = _acc9 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc10 = _acc10 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc11 = _acc11 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc12 = _acc12 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc13 = _acc13 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc14 = _acc14 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc15 = _acc15 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc16 = _acc16 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc17 = _acc17 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc18 = _acc18 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc19 = _acc19 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc20 = _acc20 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc21 = _acc21 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc22 = _acc22 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc23 = _acc23 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc24 = _acc24 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc25 = _acc25 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc26 = _acc26 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc27 = _acc27 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc28 = _acc28 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc29 = _acc29 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc30 = _acc30 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                    _acc31 = _acc31 + t0;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 16384) + (d1 * 16384) + (d4 * 128) + d3]);
                  _acc_r = _acc_r + t0;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_139(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
        const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 1 + mi;
          const _row_base: i32 = d0 * 1 + d1 * 1 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 128) + (d1 * 128) + (d2 * 128) + d4]);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t0;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 128) + (d1 * 128) + (d2 * 128) + d4]);
                  _acc_r = _acc_r + t0;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_140(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[0]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = unchecked(_in2[(d0 * 512) + (d1 * 128) + d2]);
    const t3: f32 = (t1 < t2 ? f32(1.0) : f32(0.0));
    const t4: f32 = (-t3);
    const t5: f32 = (t0 + t4);
    _out[oi] = t5;
  }
}

function kernel_141(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array, _in3: Float32Array): void {
  for (let oi: i32 = 0; oi < 65536; oi++) {
    const d0: i32 = (oi / 65536) % 1;
    const d1: i32 = (oi / 16384) % 4;
    const d2: i32 = (oi / 128) % 128;
    const d3: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[(d0 * 512) + (d1 * 128) + d2]);
    const t2: f32 = (-t1);
    const t3: f32 = unchecked(_in2[oi]);
    const t4: f32 = unchecked(_in3[(d0 * 512) + (d1 * 128) + d2]);
    const t5: f32 = (f32(1.0) / t4);
    const t6: f32 = (t3 * t5);
    const t7: f32 = (t2 * t6);
    const t8: f32 = (t0 + t7);
    _out[oi] = t8;
  }
}

function kernel_142(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 1048576; oi++) {
    const d0: i32 = (oi / 1048576) % 1;
    const d1: i32 = (oi / 262144) % 4;
    const d2: i32 = (oi / 2048) % 128;
    const d3: i32 = (oi / 128) % 16;
    const d4: i32 = oi % 128;
    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d1 * 16384) + (d2 * 128) + d4]);
    const t1: f32 = unchecked(_in1[0]);
    const t2: f32 = (t0 * t1);
    _out[oi] = t2;
  }
}

function kernel_143(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let d2: i32 = 0; d2 < 1; d2++) {
        for (let m_blk: i32 = 0; m_blk < 2; m_blk++) {
          const m_end: i32 = 16 - m_blk * 8 < 8 ? 16 - m_blk * 8 : 8;
          for (let mi: i32 = 0; mi < m_end; mi++) {
            const d3: i32 = m_blk * 8 + mi;
            const _row_base: i32 = d0 * 8192 + d1 * 2048 + d2 * 2048 + d3 * 128;
            for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
              const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
              for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
                const ni_base: i32 = n_blk * 32 + ni_grp * 32;
                let _acc0: f32 = f32(0.0);
                let _acc1: f32 = f32(0.0);
                let _acc2: f32 = f32(0.0);
                let _acc3: f32 = f32(0.0);
                let _acc4: f32 = f32(0.0);
                let _acc5: f32 = f32(0.0);
                let _acc6: f32 = f32(0.0);
                let _acc7: f32 = f32(0.0);
                let _acc8: f32 = f32(0.0);
                let _acc9: f32 = f32(0.0);
                let _acc10: f32 = f32(0.0);
                let _acc11: f32 = f32(0.0);
                let _acc12: f32 = f32(0.0);
                let _acc13: f32 = f32(0.0);
                let _acc14: f32 = f32(0.0);
                let _acc15: f32 = f32(0.0);
                let _acc16: f32 = f32(0.0);
                let _acc17: f32 = f32(0.0);
                let _acc18: f32 = f32(0.0);
                let _acc19: f32 = f32(0.0);
                let _acc20: f32 = f32(0.0);
                let _acc21: f32 = f32(0.0);
                let _acc22: f32 = f32(0.0);
                let _acc23: f32 = f32(0.0);
                let _acc24: f32 = f32(0.0);
                let _acc25: f32 = f32(0.0);
                let _acc26: f32 = f32(0.0);
                let _acc27: f32 = f32(0.0);
                let _acc28: f32 = f32(0.0);
                let _acc29: f32 = f32(0.0);
                let _acc30: f32 = f32(0.0);
                let _acc31: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    {
                      const d4: i32 = ni_base + 0;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc0 = _acc0 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 1;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc1 = _acc1 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 2;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc2 = _acc2 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 3;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc3 = _acc3 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 4;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc4 = _acc4 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 5;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc5 = _acc5 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 6;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc6 = _acc6 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 7;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc7 = _acc7 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 8;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc8 = _acc8 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 9;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc9 = _acc9 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 10;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc10 = _acc10 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 11;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc11 = _acc11 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 12;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc12 = _acc12 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 13;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc13 = _acc13 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 14;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc14 = _acc14 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 15;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc15 = _acc15 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 16;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc16 = _acc16 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 17;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc17 = _acc17 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 18;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc18 = _acc18 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 19;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc19 = _acc19 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 20;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc20 = _acc20 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 21;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc21 = _acc21 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 22;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc22 = _acc22 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 23;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc23 = _acc23 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 24;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc24 = _acc24 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 25;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc25 = _acc25 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 26;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc26 = _acc26 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 27;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc27 = _acc27 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 28;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc28 = _acc28 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 29;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc29 = _acc29 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 30;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc30 = _acc30 + t2;
                    }
                    {
                      const d4: i32 = ni_base + 31;
                      const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                      const t2: f32 = (t0 * t1);
                      _acc31 = _acc31 + t2;
                    }
                  }
                }
                unchecked(_out[_row_base + ni_base + 0] = _acc0);
                unchecked(_out[_row_base + ni_base + 1] = _acc1);
                unchecked(_out[_row_base + ni_base + 2] = _acc2);
                unchecked(_out[_row_base + ni_base + 3] = _acc3);
                unchecked(_out[_row_base + ni_base + 4] = _acc4);
                unchecked(_out[_row_base + ni_base + 5] = _acc5);
                unchecked(_out[_row_base + ni_base + 6] = _acc6);
                unchecked(_out[_row_base + ni_base + 7] = _acc7);
                unchecked(_out[_row_base + ni_base + 8] = _acc8);
                unchecked(_out[_row_base + ni_base + 9] = _acc9);
                unchecked(_out[_row_base + ni_base + 10] = _acc10);
                unchecked(_out[_row_base + ni_base + 11] = _acc11);
                unchecked(_out[_row_base + ni_base + 12] = _acc12);
                unchecked(_out[_row_base + ni_base + 13] = _acc13);
                unchecked(_out[_row_base + ni_base + 14] = _acc14);
                unchecked(_out[_row_base + ni_base + 15] = _acc15);
                unchecked(_out[_row_base + ni_base + 16] = _acc16);
                unchecked(_out[_row_base + ni_base + 17] = _acc17);
                unchecked(_out[_row_base + ni_base + 18] = _acc18);
                unchecked(_out[_row_base + ni_base + 19] = _acc19);
                unchecked(_out[_row_base + ni_base + 20] = _acc20);
                unchecked(_out[_row_base + ni_base + 21] = _acc21);
                unchecked(_out[_row_base + ni_base + 22] = _acc22);
                unchecked(_out[_row_base + ni_base + 23] = _acc23);
                unchecked(_out[_row_base + ni_base + 24] = _acc24);
                unchecked(_out[_row_base + ni_base + 25] = _acc25);
                unchecked(_out[_row_base + ni_base + 26] = _acc26);
                unchecked(_out[_row_base + ni_base + 27] = _acc27);
                unchecked(_out[_row_base + ni_base + 28] = _acc28);
                unchecked(_out[_row_base + ni_base + 29] = _acc29);
                unchecked(_out[_row_base + ni_base + 30] = _acc30);
                unchecked(_out[_row_base + ni_base + 31] = _acc31);
              }
              for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
                const d4: i32 = n_blk * 32 + ni;
                let _acc_r: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d5 * 2048) + (d3 * 128) + d4]);
                    const t2: f32 = (t0 * t1);
                    _acc_r = _acc_r + t2;
                  }
                }
                unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
              }
            }
          }
        }
      }
    }
  }
}

function kernel_144(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 4; d1++) {
      for (let d2: i32 = 0; d2 < 128; d2++) {
        for (let m_blk: i32 = 0; m_blk < 2; m_blk++) {
          const m_end: i32 = 16 - m_blk * 8 < 8 ? 16 - m_blk * 8 : 8;
          for (let mi: i32 = 0; mi < m_end; mi++) {
            const d3: i32 = m_blk * 8 + mi;
            const _row_base: i32 = d0 * 8192 + d1 * 2048 + d2 * 16 + d3 * 1;
            for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
              const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
              for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
                const ni_base: i32 = n_blk * 1 + ni_grp * 1;
                let _acc0: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 128) + d5]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 128) + d5]);
                    const t2: f32 = (t0 * t1);
                    {
                      const d4: i32 = ni_base + 0;
                      _acc0 = _acc0 + t2;
                    }
                  }
                }
                unchecked(_out[_row_base + ni_base + 0] = _acc0);
              }
              for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
                const d4: i32 = n_blk * 1 + ni;
                let _acc_r: f32 = f32(0.0);
                for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                  const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                  for (let ki: i32 = 0; ki < k_end; ki++) {
                    const d5: i32 = k_blk * 32 + ki;
                    const t0: f32 = unchecked(_in0[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 128) + d5]);
                    const t1: f32 = unchecked(_in1[(d0 * 1048576) + (d1 * 262144) + (d2 * 2048) + (d3 * 128) + d5]);
                    const t2: f32 = (t0 * t1);
                    _acc_r = _acc_r + t2;
                  }
                }
                unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
              }
            }
          }
        }
      }
    }
  }
}

function kernel_145(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = (oi / 64) % 1;
    const d3: i32 = (oi / 16) % 4;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 2048) + (d1 * 16) + d4]);
    _out[oi] = t0;
  }
}

function kernel_146(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 8192) % 1;
    const d1: i32 = (oi / 64) % 128;
    const d2: i32 = (oi / 64) % 1;
    const d3: i32 = (oi / 16) % 4;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d3 * 2048) + (d4 * 128) + d1]);
    _out[oi] = t0;
  }
}

function kernel_147(_out: Float32Array, _in0: Float32Array, _in1: Float32Array, _in2: Float32Array): void {
  for (let oi: i32 = 0; oi < 24576; oi++) {
    const d0: i32 = (oi / 24576) % 1;
    const d1: i32 = (oi / 192) % 128;
    const d2: i32 = (oi / 64) % 3;
    const d3: i32 = (oi / 16) % 4;
    const d4: i32 = oi % 16;
    const t0: f32 = unchecked(_in0[oi]);
    const t1: f32 = unchecked(_in1[oi]);
    const t2: f32 = (t0 + t1);
    const t3: f32 = unchecked(_in2[oi]);
    const t4: f32 = (t2 + t3);
    _out[oi] = t4;
  }
}

function kernel_148(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 24576; oi++) {
    const d0: i32 = (oi / 24576) % 1;
    const d1: i32 = (oi / 192) % 128;
    const d2: i32 = oi % 192;
    const t0: f32 = unchecked(_in0[(d0 * 24576) + (d1 * 192) + d2]);
    _out[oi] = t0;
  }
}

function kernel_149(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {
      const m_end: i32 = 1 - m_blk * 1 < 1 ? 1 - m_blk * 1 : 1;
      for (let mi: i32 = 0; mi < m_end; mi++) {
        const d1: i32 = m_blk * 1 + mi;
        const _row_base: i32 = d0 * 192 + d1 * 192;
        for (let n_blk: i32 = 0; n_blk < 6; n_blk++) {
          const n_tile: i32 = 192 - n_blk * 32 < 32 ? 192 - n_blk * 32 : 32;
          for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
            const ni_base: i32 = n_blk * 32 + ni_grp * 32;
            let _acc0: f32 = f32(0.0);
            let _acc1: f32 = f32(0.0);
            let _acc2: f32 = f32(0.0);
            let _acc3: f32 = f32(0.0);
            let _acc4: f32 = f32(0.0);
            let _acc5: f32 = f32(0.0);
            let _acc6: f32 = f32(0.0);
            let _acc7: f32 = f32(0.0);
            let _acc8: f32 = f32(0.0);
            let _acc9: f32 = f32(0.0);
            let _acc10: f32 = f32(0.0);
            let _acc11: f32 = f32(0.0);
            let _acc12: f32 = f32(0.0);
            let _acc13: f32 = f32(0.0);
            let _acc14: f32 = f32(0.0);
            let _acc15: f32 = f32(0.0);
            let _acc16: f32 = f32(0.0);
            let _acc17: f32 = f32(0.0);
            let _acc18: f32 = f32(0.0);
            let _acc19: f32 = f32(0.0);
            let _acc20: f32 = f32(0.0);
            let _acc21: f32 = f32(0.0);
            let _acc22: f32 = f32(0.0);
            let _acc23: f32 = f32(0.0);
            let _acc24: f32 = f32(0.0);
            let _acc25: f32 = f32(0.0);
            let _acc26: f32 = f32(0.0);
            let _acc27: f32 = f32(0.0);
            let _acc28: f32 = f32(0.0);
            let _acc29: f32 = f32(0.0);
            let _acc30: f32 = f32(0.0);
            let _acc31: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                {
                  const d2: i32 = ni_base + 0;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc0 = _acc0 + t0;
                }
                {
                  const d2: i32 = ni_base + 1;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc1 = _acc1 + t0;
                }
                {
                  const d2: i32 = ni_base + 2;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc2 = _acc2 + t0;
                }
                {
                  const d2: i32 = ni_base + 3;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc3 = _acc3 + t0;
                }
                {
                  const d2: i32 = ni_base + 4;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc4 = _acc4 + t0;
                }
                {
                  const d2: i32 = ni_base + 5;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc5 = _acc5 + t0;
                }
                {
                  const d2: i32 = ni_base + 6;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc6 = _acc6 + t0;
                }
                {
                  const d2: i32 = ni_base + 7;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc7 = _acc7 + t0;
                }
                {
                  const d2: i32 = ni_base + 8;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc8 = _acc8 + t0;
                }
                {
                  const d2: i32 = ni_base + 9;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc9 = _acc9 + t0;
                }
                {
                  const d2: i32 = ni_base + 10;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc10 = _acc10 + t0;
                }
                {
                  const d2: i32 = ni_base + 11;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc11 = _acc11 + t0;
                }
                {
                  const d2: i32 = ni_base + 12;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc12 = _acc12 + t0;
                }
                {
                  const d2: i32 = ni_base + 13;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc13 = _acc13 + t0;
                }
                {
                  const d2: i32 = ni_base + 14;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc14 = _acc14 + t0;
                }
                {
                  const d2: i32 = ni_base + 15;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc15 = _acc15 + t0;
                }
                {
                  const d2: i32 = ni_base + 16;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc16 = _acc16 + t0;
                }
                {
                  const d2: i32 = ni_base + 17;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc17 = _acc17 + t0;
                }
                {
                  const d2: i32 = ni_base + 18;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc18 = _acc18 + t0;
                }
                {
                  const d2: i32 = ni_base + 19;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc19 = _acc19 + t0;
                }
                {
                  const d2: i32 = ni_base + 20;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc20 = _acc20 + t0;
                }
                {
                  const d2: i32 = ni_base + 21;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc21 = _acc21 + t0;
                }
                {
                  const d2: i32 = ni_base + 22;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc22 = _acc22 + t0;
                }
                {
                  const d2: i32 = ni_base + 23;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc23 = _acc23 + t0;
                }
                {
                  const d2: i32 = ni_base + 24;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc24 = _acc24 + t0;
                }
                {
                  const d2: i32 = ni_base + 25;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc25 = _acc25 + t0;
                }
                {
                  const d2: i32 = ni_base + 26;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc26 = _acc26 + t0;
                }
                {
                  const d2: i32 = ni_base + 27;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc27 = _acc27 + t0;
                }
                {
                  const d2: i32 = ni_base + 28;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc28 = _acc28 + t0;
                }
                {
                  const d2: i32 = ni_base + 29;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc29 = _acc29 + t0;
                }
                {
                  const d2: i32 = ni_base + 30;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc30 = _acc30 + t0;
                }
                {
                  const d2: i32 = ni_base + 31;
                  const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                  _acc31 = _acc31 + t0;
                }
              }
            }
            unchecked(_out[_row_base + ni_base + 0] = _acc0);
            unchecked(_out[_row_base + ni_base + 1] = _acc1);
            unchecked(_out[_row_base + ni_base + 2] = _acc2);
            unchecked(_out[_row_base + ni_base + 3] = _acc3);
            unchecked(_out[_row_base + ni_base + 4] = _acc4);
            unchecked(_out[_row_base + ni_base + 5] = _acc5);
            unchecked(_out[_row_base + ni_base + 6] = _acc6);
            unchecked(_out[_row_base + ni_base + 7] = _acc7);
            unchecked(_out[_row_base + ni_base + 8] = _acc8);
            unchecked(_out[_row_base + ni_base + 9] = _acc9);
            unchecked(_out[_row_base + ni_base + 10] = _acc10);
            unchecked(_out[_row_base + ni_base + 11] = _acc11);
            unchecked(_out[_row_base + ni_base + 12] = _acc12);
            unchecked(_out[_row_base + ni_base + 13] = _acc13);
            unchecked(_out[_row_base + ni_base + 14] = _acc14);
            unchecked(_out[_row_base + ni_base + 15] = _acc15);
            unchecked(_out[_row_base + ni_base + 16] = _acc16);
            unchecked(_out[_row_base + ni_base + 17] = _acc17);
            unchecked(_out[_row_base + ni_base + 18] = _acc18);
            unchecked(_out[_row_base + ni_base + 19] = _acc19);
            unchecked(_out[_row_base + ni_base + 20] = _acc20);
            unchecked(_out[_row_base + ni_base + 21] = _acc21);
            unchecked(_out[_row_base + ni_base + 22] = _acc22);
            unchecked(_out[_row_base + ni_base + 23] = _acc23);
            unchecked(_out[_row_base + ni_base + 24] = _acc24);
            unchecked(_out[_row_base + ni_base + 25] = _acc25);
            unchecked(_out[_row_base + ni_base + 26] = _acc26);
            unchecked(_out[_row_base + ni_base + 27] = _acc27);
            unchecked(_out[_row_base + ni_base + 28] = _acc28);
            unchecked(_out[_row_base + ni_base + 29] = _acc29);
            unchecked(_out[_row_base + ni_base + 30] = _acc30);
            unchecked(_out[_row_base + ni_base + 31] = _acc31);
          }
          for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
            const d2: i32 = n_blk * 32 + ni;
            let _acc_r: f32 = f32(0.0);
            for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
              const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
              for (let ki: i32 = 0; ki < k_end; ki++) {
                const d3: i32 = k_blk * 32 + ki;
                const t0: f32 = unchecked(_in0[(d0 * 24576) + (d3 * 192) + d2]);
                _acc_r = _acc_r + t0;
              }
            }
            unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
          }
        }
      }
    }
  }
}

function kernel_150(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 192; oi++) {
    const d0: i32 = oi % 192;
    const t0: f32 = unchecked(_in0[d0]);
    _out[oi] = t0;
  }
}

function kernel_151(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 1572864; oi++) {
    const d0: i32 = (oi / 1572864) % 1;
    const d1: i32 = (oi / 12288) % 128;
    const d2: i32 = (oi / 192) % 64;
    const d3: i32 = oi % 192;
    const t0: f32 = unchecked(_in0[(d0 * 24576) + (d1 * 192) + d3]);
    _out[oi] = t0;
  }
}

function kernel_152(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 12288 + d1 * 12288 + d2 * 192;
          for (let n_blk: i32 = 0; n_blk < 6; n_blk++) {
            const n_tile: i32 = 192 - n_blk * 32 < 32 ? 192 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d4 * 12288) + (d2 * 192) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_153(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 8; m_blk++) {
        const m_end: i32 = 64 - m_blk * 8 < 8 ? 64 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 8192 + d1 * 64 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 6; k_blk++) {
                const k_end: i32 = 192 - k_blk * 32 < 32 ? 192 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d2 * 192) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d2 * 192) + d4]);
                  const t2: f32 = (t0 * t1);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 6; k_blk++) {
                const k_end: i32 = 192 - k_blk * 32 < 32 ? 192 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 1572864) + (d1 * 12288) + (d2 * 192) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 1572864) + (d1 * 12288) + (d2 * 192) + d4]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_154(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 12288; oi++) {
    const d0: i32 = (oi / 192) % 64;
    const d1: i32 = oi % 192;
    const t0: f32 = unchecked(_in0[(d0 * 192) + d1]);
    _out[oi] = t0;
  }
}

function kernel_155(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 16384 + d1 * 16384 + d2 * 128;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 4 - k_blk * 4 < 4 ? 4 - k_blk * 4 : 4;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 4 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 + t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 4 - k_blk * 4 < 4 ? 4 - k_blk * 4 : 4;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 4 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 65536) + (d4 * 16384) + (d2 * 128) + d3]);
                  const t2: f32 = (t0 + t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_156(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 16384 + d1 * 16384 + d2 * 128;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 1 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 1 + ki;
                  const t0: f32 = unchecked(_in0[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                  const t1: f32 = unchecked(_in1[(d4 * 16384) + (d1 * 16384) + (d2 * 128) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_157(_out: Float32Array, _in0: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 16; m_blk++) {
        const m_end: i32 = 128 - m_blk * 8 < 8 ? 128 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 16384 + d1 * 16384 + d2 * 128;
          for (let n_blk: i32 = 0; n_blk < 4; n_blk++) {
            const n_tile: i32 = 128 - n_blk * 32 < 32 ? 128 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 1 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc0 = _acc0 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc1 = _acc1 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc2 = _acc2 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc3 = _acc3 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc4 = _acc4 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc5 = _acc5 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc6 = _acc6 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc7 = _acc7 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc8 = _acc8 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc9 = _acc9 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc10 = _acc10 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc11 = _acc11 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc12 = _acc12 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc13 = _acc13 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc14 = _acc14 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc15 = _acc15 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc16 = _acc16 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc17 = _acc17 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc18 = _acc18 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc19 = _acc19 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc20 = _acc20 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc21 = _acc21 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc22 = _acc22 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc23 = _acc23 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc24 = _acc24 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc25 = _acc25 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc26 = _acc26 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc27 = _acc27 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc28 = _acc28 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc29 = _acc29 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc30 = _acc30 + t0;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                    _acc31 = _acc31 + t0;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 1; k_blk++) {
                const k_end: i32 = 1 - k_blk * 1 < 1 ? 1 - k_blk * 1 : 1;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 1 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 16384) + (d4 * 16384) + (d2 * 128) + d3]);
                  _acc_r = _acc_r + t0;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_158(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 8192; oi++) {
    const d0: i32 = (oi / 64) % 128;
    const d1: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 64) + d1]);
    _out[oi] = t0;
  }
}

function kernel_159(_out: Float32Array, _in0: Float32Array): void {
  for (let oi: i32 = 0; oi < 811008; oi++) {
    const d0: i32 = (oi / 811008) % 1;
    const d1: i32 = (oi / 6336) % 128;
    const d2: i32 = (oi / 64) % 99;
    const d3: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d0 * 8192) + (d1 * 64) + d3]);
    _out[oi] = t0;
  }
}

function kernel_160(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 1; d1++) {
      for (let m_blk: i32 = 0; m_blk < 13; m_blk++) {
        const m_end: i32 = 99 - m_blk * 8 < 8 ? 99 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 6336 + d1 * 6336 + d2 * 64;
          for (let n_blk: i32 = 0; n_blk < 2; n_blk++) {
            const n_tile: i32 = 64 - n_blk * 32 < 32 ? 64 - n_blk * 32 : 32;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 32); ni_grp++) {
              const ni_base: i32 = n_blk * 32 + ni_grp * 32;
              let _acc0: f32 = f32(0.0);
              let _acc1: f32 = f32(0.0);
              let _acc2: f32 = f32(0.0);
              let _acc3: f32 = f32(0.0);
              let _acc4: f32 = f32(0.0);
              let _acc5: f32 = f32(0.0);
              let _acc6: f32 = f32(0.0);
              let _acc7: f32 = f32(0.0);
              let _acc8: f32 = f32(0.0);
              let _acc9: f32 = f32(0.0);
              let _acc10: f32 = f32(0.0);
              let _acc11: f32 = f32(0.0);
              let _acc12: f32 = f32(0.0);
              let _acc13: f32 = f32(0.0);
              let _acc14: f32 = f32(0.0);
              let _acc15: f32 = f32(0.0);
              let _acc16: f32 = f32(0.0);
              let _acc17: f32 = f32(0.0);
              let _acc18: f32 = f32(0.0);
              let _acc19: f32 = f32(0.0);
              let _acc20: f32 = f32(0.0);
              let _acc21: f32 = f32(0.0);
              let _acc22: f32 = f32(0.0);
              let _acc23: f32 = f32(0.0);
              let _acc24: f32 = f32(0.0);
              let _acc25: f32 = f32(0.0);
              let _acc26: f32 = f32(0.0);
              let _acc27: f32 = f32(0.0);
              let _acc28: f32 = f32(0.0);
              let _acc29: f32 = f32(0.0);
              let _acc30: f32 = f32(0.0);
              let _acc31: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  {
                    const d3: i32 = ni_base + 0;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc0 = _acc0 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 1;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc1 = _acc1 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 2;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc2 = _acc2 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 3;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc3 = _acc3 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 4;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc4 = _acc4 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 5;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc5 = _acc5 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 6;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc6 = _acc6 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 7;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc7 = _acc7 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 8;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc8 = _acc8 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 9;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc9 = _acc9 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 10;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc10 = _acc10 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 11;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc11 = _acc11 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 12;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc12 = _acc12 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 13;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc13 = _acc13 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 14;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc14 = _acc14 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 15;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc15 = _acc15 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 16;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc16 = _acc16 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 17;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc17 = _acc17 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 18;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc18 = _acc18 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 19;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc19 = _acc19 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 20;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc20 = _acc20 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 21;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc21 = _acc21 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 22;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc22 = _acc22 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 23;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc23 = _acc23 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 24;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc24 = _acc24 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 25;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc25 = _acc25 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 26;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc26 = _acc26 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 27;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc27 = _acc27 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 28;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc28 = _acc28 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 29;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc29 = _acc29 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 30;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc30 = _acc30 + t2;
                  }
                  {
                    const d3: i32 = ni_base + 31;
                    const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                    const t2: f32 = (t0 * t1);
                    _acc31 = _acc31 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
              unchecked(_out[_row_base + ni_base + 1] = _acc1);
              unchecked(_out[_row_base + ni_base + 2] = _acc2);
              unchecked(_out[_row_base + ni_base + 3] = _acc3);
              unchecked(_out[_row_base + ni_base + 4] = _acc4);
              unchecked(_out[_row_base + ni_base + 5] = _acc5);
              unchecked(_out[_row_base + ni_base + 6] = _acc6);
              unchecked(_out[_row_base + ni_base + 7] = _acc7);
              unchecked(_out[_row_base + ni_base + 8] = _acc8);
              unchecked(_out[_row_base + ni_base + 9] = _acc9);
              unchecked(_out[_row_base + ni_base + 10] = _acc10);
              unchecked(_out[_row_base + ni_base + 11] = _acc11);
              unchecked(_out[_row_base + ni_base + 12] = _acc12);
              unchecked(_out[_row_base + ni_base + 13] = _acc13);
              unchecked(_out[_row_base + ni_base + 14] = _acc14);
              unchecked(_out[_row_base + ni_base + 15] = _acc15);
              unchecked(_out[_row_base + ni_base + 16] = _acc16);
              unchecked(_out[_row_base + ni_base + 17] = _acc17);
              unchecked(_out[_row_base + ni_base + 18] = _acc18);
              unchecked(_out[_row_base + ni_base + 19] = _acc19);
              unchecked(_out[_row_base + ni_base + 20] = _acc20);
              unchecked(_out[_row_base + ni_base + 21] = _acc21);
              unchecked(_out[_row_base + ni_base + 22] = _acc22);
              unchecked(_out[_row_base + ni_base + 23] = _acc23);
              unchecked(_out[_row_base + ni_base + 24] = _acc24);
              unchecked(_out[_row_base + ni_base + 25] = _acc25);
              unchecked(_out[_row_base + ni_base + 26] = _acc26);
              unchecked(_out[_row_base + ni_base + 27] = _acc27);
              unchecked(_out[_row_base + ni_base + 28] = _acc28);
              unchecked(_out[_row_base + ni_base + 29] = _acc29);
              unchecked(_out[_row_base + ni_base + 30] = _acc30);
              unchecked(_out[_row_base + ni_base + 31] = _acc31);
            }
            for (let ni: i32 = (n_tile / 32) * 32; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 32 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 4; k_blk++) {
                const k_end: i32 = 128 - k_blk * 32 < 32 ? 128 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d4 * 6336) + (d2 * 64) + d3]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 32 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_161(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let d0: i32 = 0; d0 < 1; d0++) {
    for (let d1: i32 = 0; d1 < 128; d1++) {
      for (let m_blk: i32 = 0; m_blk < 13; m_blk++) {
        const m_end: i32 = 99 - m_blk * 8 < 8 ? 99 - m_blk * 8 : 8;
        for (let mi: i32 = 0; mi < m_end; mi++) {
          const d2: i32 = m_blk * 8 + mi;
          const _row_base: i32 = d0 * 12672 + d1 * 99 + d2 * 1;
          for (let n_blk: i32 = 0; n_blk < 1; n_blk++) {
            const n_tile: i32 = 1 - n_blk * 1 < 1 ? 1 - n_blk * 1 : 1;
            for (let ni_grp: i32 = 0; ni_grp < (n_tile / 1); ni_grp++) {
              const ni_base: i32 = n_blk * 1 + ni_grp * 1;
              let _acc0: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d2 * 64) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d2 * 64) + d4]);
                  const t2: f32 = (t0 * t1);
                  {
                    const d3: i32 = ni_base + 0;
                    _acc0 = _acc0 + t2;
                  }
                }
              }
              unchecked(_out[_row_base + ni_base + 0] = _acc0);
            }
            for (let ni: i32 = (n_tile / 1) * 1; ni < n_tile; ni++) {
              const d3: i32 = n_blk * 1 + ni;
              let _acc_r: f32 = f32(0.0);
              for (let k_blk: i32 = 0; k_blk < 2; k_blk++) {
                const k_end: i32 = 64 - k_blk * 32 < 32 ? 64 - k_blk * 32 : 32;
                for (let ki: i32 = 0; ki < k_end; ki++) {
                  const d4: i32 = k_blk * 32 + ki;
                  const t0: f32 = unchecked(_in0[(d0 * 811008) + (d1 * 6336) + (d2 * 64) + d4]);
                  const t1: f32 = unchecked(_in1[(d0 * 811008) + (d1 * 6336) + (d2 * 64) + d4]);
                  const t2: f32 = (t0 * t1);
                  _acc_r = _acc_r + t2;
                }
              }
              unchecked(_out[_row_base + n_blk * 1 + ni] = _acc_r);
            }
          }
        }
      }
    }
  }
}

function kernel_162(_out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
  for (let oi: i32 = 0; oi < 6336; oi++) {
    const d0: i32 = (oi / 64) % 99;
    const d1: i32 = oi % 64;
    const t0: f32 = unchecked(_in0[(d1 * 99) + d0]);
    const t1: f32 = unchecked(_in1[(d0 * 64) + d1]);
    const t2: f32 = (t0 + t1);
    _out[oi] = t2;
  }
}

export function execute(input_0: Float32Array, input_1: Float32Array, input_2: Float32Array, input_3: Float32Array, input_4: Float32Array, input_5: Float32Array, input_6: Float32Array, input_7: Float32Array, input_8: Float32Array, input_9: Float32Array, input_10: Float32Array, input_11: Float32Array, input_12: Float32Array, input_13: Float32Array, input_14: Float32Array, input_15: Float32Array, input_16: Float32Array, input_17: Float32Array, input_18: Float32Array, input_19: Float32Array, input_20: Float32Array, input_21: Float32Array, input_22: Float32Array, input_23: Float32Array, input_24: Float32Array, input_25: Float32Array, input_26: Float32Array, input_27: Float32Array, input_28: Float32Array, targets: Float32Array): Float32Array {
  const buf0 = input_0;
  const buf1 = input_1;
  const buf2 = input_2;
  const buf3 = new Float32Array(99);
  for (let i: i32 = 0; i < 99; i++) buf3[i] = f32(i);
  const buf5 = new Float32Array(12672);
  kernel_0(buf5, buf3);
  const buf7 = new Float32Array(12672);
  kernel_1(buf7, buf0);
  const buf8 = new Float32Array(1);
  buf8[0] = f32(0.5);
  const buf11 = new Float32Array(1);
  buf11[0] = f32(0.5);
  const buf13 = new Float32Array(12672);
  kernel_2(buf13, buf5, buf8, buf7);
  const buf14 = new Float32Array(12672);
  kernel_3(buf14, buf7, buf5, buf11);
  const buf18 = new Float32Array(811008);
  kernel_4(buf18, buf13, buf14);
  const buf19 = new Float32Array(811008);
  kernel_5(buf19, buf1);
  const buf21 = new Float32Array(8192);
  kernel_6(buf21, buf18, buf19);
  const buf25 = new Float32Array(8192);
  kernel_7(buf25, buf21, buf2);
  const buf26 = new Float32Array(128);
  for (let i: i32 = 0; i < 128; i++) buf26[i] = f32(i);
  const buf29 = new Float32Array(128);
  for (let i: i32 = 0; i < 128; i++) buf29[i] = f32(i);
  const buf32 = new Float32Array(16384);
  kernel_8(buf32, buf26, buf29);
  const buf33 = new Float32Array(1);
  buf33[0] = f32(1000000.0);
  const buf34 = new Float32Array(1);
  kernel_9(buf34, buf33);
  const buf36 = new Float32Array(65536);
  kernel_10(buf36, buf32, buf34);
  const buf37 = input_3;
  const buf38 = input_4;
  const buf39 = input_5;
  const buf40 = input_6;
  const buf41 = input_7;
  const buf42 = input_8;
  const buf43 = input_9;
  const buf44 = input_10;
  const buf45 = input_11;
  const buf46 = input_12;
  const buf47 = input_13;
  const buf48 = input_14;
  const buf49 = new Float32Array(128);
  kernel_11(buf49, buf25);
  const buf50 = new Float32Array(1);
  buf50[0] = f32(0.015625);
  const buf53 = new Float32Array(8192);
  kernel_12(buf53, buf25, buf49, buf50);
  const buf55 = new Float32Array(128);
  kernel_13(buf55, buf53);
  const buf56 = new Float32Array(1);
  buf56[0] = f32(0.015625);
  const buf58 = new Float32Array(1);
  buf58[0] = f32(0.00001);
  const buf59 = new Float32Array(128);
  kernel_14(buf59, buf55, buf56, buf58);
  const buf60 = new Float32Array(128);
  kernel_15(buf60, buf59);
  const buf61 = new Float32Array(128);
  kernel_16(buf61, buf60);
  const buf62 = new Float32Array(8192);
  kernel_17(buf62, buf53, buf61);
  const buf67 = new Float32Array(1572864);
  kernel_18(buf67, buf62, buf37, buf38);
  const buf68 = new Float32Array(1572864);
  kernel_19(buf68, buf39);
  const buf70 = new Float32Array(24576);
  kernel_20(buf70, buf67, buf68);
  const buf72 = new Float32Array(24576);
  kernel_21(buf72, buf70, buf40);
  const buf73 = new Float32Array(24576);
  kernel_22(buf73, buf72);
  const buf74 = new Float32Array(8192);
  kernel_23(buf74, buf73);
  const buf76 = new Float32Array(8192);
  kernel_24(buf76, buf73);
  const buf78 = new Float32Array(8192);
  kernel_25(buf78, buf73);
  const buf86 = new Float32Array(1048576);
  kernel_26(buf86, buf74);
  const buf87 = new Float32Array(1048576);
  kernel_27(buf87, buf76);
  const buf89 = new Float32Array(65536);
  kernel_28(buf89, buf86, buf87);
  const buf90 = new Float32Array(65536);
  kernel_29(buf90, buf89);
  const buf91 = new Float32Array(1);
  buf91[0] = f32(0.25);
  const buf93 = new Float32Array(65536);
  kernel_30(buf93, buf90, buf91, buf36);
  const buf94 = new Float32Array(512);
  kernel_31(buf94, buf93);
  const buf96 = new Float32Array(65536);
  kernel_32(buf96, buf93, buf94);
  const buf97 = new Float32Array(1);
  buf97[0] = f32(1.4426950408889634);
  const buf98 = new Float32Array(65536);
  kernel_33(buf98, buf96, buf97);
  const buf99 = new Float32Array(65536);
  kernel_34(buf99, buf98);
  const buf100 = new Float32Array(512);
  kernel_35(buf100, buf99);
  const buf101 = new Float32Array(512);
  kernel_36(buf101, buf100);
  const buf105 = new Float32Array(1048576);
  kernel_37(buf105, buf101, buf99);
  const buf106 = new Float32Array(1048576);
  kernel_38(buf106, buf78);
  const buf108 = new Float32Array(8192);
  kernel_39(buf108, buf105, buf106);
  const buf110 = new Float32Array(8192);
  kernel_40(buf110, buf108);
  const buf111 = new Float32Array(8192);
  kernel_41(buf111, buf110);
  const buf114 = new Float32Array(524288);
  kernel_42(buf114, buf111);
  const buf115 = new Float32Array(524288);
  kernel_43(buf115, buf41);
  const buf117 = new Float32Array(8192);
  kernel_44(buf117, buf114, buf115);
  const buf120 = new Float32Array(8192);
  kernel_45(buf120, buf25, buf117, buf42);
  const buf121 = new Float32Array(128);
  kernel_11(buf121, buf120);
  const buf122 = new Float32Array(1);
  buf122[0] = f32(0.015625);
  const buf125 = new Float32Array(8192);
  kernel_12(buf125, buf120, buf121, buf122);
  const buf127 = new Float32Array(128);
  kernel_13(buf127, buf125);
  const buf128 = new Float32Array(1);
  buf128[0] = f32(0.015625);
  const buf130 = new Float32Array(1);
  buf130[0] = f32(0.00001);
  const buf131 = new Float32Array(128);
  kernel_14(buf131, buf127, buf128, buf130);
  const buf132 = new Float32Array(128);
  kernel_15(buf132, buf131);
  const buf133 = new Float32Array(128);
  kernel_16(buf133, buf132);
  const buf134 = new Float32Array(8192);
  kernel_17(buf134, buf125, buf133);
  const buf139 = new Float32Array(2097152);
  kernel_46(buf139, buf134, buf43, buf44);
  const buf140 = new Float32Array(2097152);
  kernel_47(buf140, buf45);
  const buf142 = new Float32Array(32768);
  kernel_48(buf142, buf139, buf140);
  const buf144 = new Float32Array(32768);
  kernel_49(buf144, buf142, buf46);
  const buf145 = new Float32Array(32768);
  kernel_50(buf145, buf144);
  const buf146 = new Float32Array(32768);
  kernel_51(buf146, buf145, buf144);
  const buf147 = new Float32Array(1);
  buf147[0] = f32(0.7978845608028654);
  const buf148 = new Float32Array(1);
  buf148[0] = f32(0.044715);
  const buf150 = new Float32Array(32768);
  kernel_52(buf150, buf144, buf148, buf146);
  const buf152 = new Float32Array(1);
  buf152[0] = f32(10.0);
  const buf153 = new Float32Array(1);
  kernel_9(buf153, buf152);
  const buf154 = new Float32Array(1);
  buf154[0] = f32(10.0);
  const buf155 = new Float32Array(32768);
  kernel_53(buf155, buf147, buf150);
  const buf156 = new Float32Array(1);
  kernel_9(buf156, buf154);
  const buf158 = new Float32Array(32768);
  kernel_54(buf158, buf155, buf156);
  const buf159 = new Float32Array(32768);
  kernel_55(buf159, buf158, buf153);
  const buf160 = new Float32Array(1);
  buf160[0] = f32(2.0);
  const buf161 = new Float32Array(32768);
  kernel_56(buf161, buf159, buf160);
  const buf162 = new Float32Array(1);
  buf162[0] = f32(1.4426950408889634);
  const buf163 = new Float32Array(32768);
  kernel_56(buf163, buf161, buf162);
  const buf164 = new Float32Array(32768);
  kernel_57(buf164, buf163);
  const buf165 = new Float32Array(1);
  buf165[0] = f32(1.0);
  const buf167 = new Float32Array(32768);
  kernel_58(buf167, buf164, buf165);
  const buf168 = new Float32Array(1);
  buf168[0] = f32(1.0);
  const buf169 = new Float32Array(32768);
  kernel_59(buf169, buf164, buf168);
  const buf170 = new Float32Array(32768);
  kernel_60(buf170, buf169);
  const buf172 = new Float32Array(1);
  buf172[0] = f32(0.5);
  const buf173 = new Float32Array(32768);
  kernel_61(buf173, buf172, buf144);
  const buf174 = new Float32Array(1);
  buf174[0] = f32(1.0);
  const buf175 = new Float32Array(32768);
  kernel_62(buf175, buf174, buf167, buf170);
  const buf179 = new Float32Array(2097152);
  kernel_63(buf179, buf173, buf175);
  const buf180 = new Float32Array(2097152);
  kernel_64(buf180, buf47);
  const buf182 = new Float32Array(8192);
  kernel_65(buf182, buf179, buf180);
  const buf185 = new Float32Array(8192);
  kernel_45(buf185, buf120, buf182, buf48);
  const buf186 = input_15;
  const buf187 = input_16;
  const buf188 = input_17;
  const buf189 = input_18;
  const buf190 = input_19;
  const buf191 = input_20;
  const buf192 = input_21;
  const buf193 = input_22;
  const buf194 = input_23;
  const buf195 = input_24;
  const buf196 = input_25;
  const buf197 = input_26;
  const buf198 = new Float32Array(128);
  kernel_11(buf198, buf185);
  const buf199 = new Float32Array(1);
  buf199[0] = f32(0.015625);
  const buf202 = new Float32Array(8192);
  kernel_12(buf202, buf185, buf198, buf199);
  const buf204 = new Float32Array(128);
  kernel_13(buf204, buf202);
  const buf205 = new Float32Array(1);
  buf205[0] = f32(0.015625);
  const buf207 = new Float32Array(1);
  buf207[0] = f32(0.00001);
  const buf208 = new Float32Array(128);
  kernel_14(buf208, buf204, buf205, buf207);
  const buf209 = new Float32Array(128);
  kernel_15(buf209, buf208);
  const buf210 = new Float32Array(128);
  kernel_16(buf210, buf209);
  const buf211 = new Float32Array(8192);
  kernel_17(buf211, buf202, buf210);
  const buf216 = new Float32Array(1572864);
  kernel_18(buf216, buf211, buf186, buf187);
  const buf217 = new Float32Array(1572864);
  kernel_19(buf217, buf188);
  const buf219 = new Float32Array(24576);
  kernel_20(buf219, buf216, buf217);
  const buf221 = new Float32Array(24576);
  kernel_21(buf221, buf219, buf189);
  const buf222 = new Float32Array(24576);
  kernel_22(buf222, buf221);
  const buf223 = new Float32Array(8192);
  kernel_23(buf223, buf222);
  const buf225 = new Float32Array(8192);
  kernel_24(buf225, buf222);
  const buf227 = new Float32Array(8192);
  kernel_25(buf227, buf222);
  const buf235 = new Float32Array(1048576);
  kernel_26(buf235, buf223);
  const buf236 = new Float32Array(1048576);
  kernel_27(buf236, buf225);
  const buf238 = new Float32Array(65536);
  kernel_28(buf238, buf235, buf236);
  const buf239 = new Float32Array(65536);
  kernel_29(buf239, buf238);
  const buf240 = new Float32Array(1);
  buf240[0] = f32(0.25);
  const buf242 = new Float32Array(65536);
  kernel_30(buf242, buf239, buf240, buf36);
  const buf243 = new Float32Array(512);
  kernel_31(buf243, buf242);
  const buf245 = new Float32Array(65536);
  kernel_32(buf245, buf242, buf243);
  const buf246 = new Float32Array(1);
  buf246[0] = f32(1.4426950408889634);
  const buf247 = new Float32Array(65536);
  kernel_33(buf247, buf245, buf246);
  const buf248 = new Float32Array(65536);
  kernel_34(buf248, buf247);
  const buf249 = new Float32Array(512);
  kernel_35(buf249, buf248);
  const buf250 = new Float32Array(512);
  kernel_36(buf250, buf249);
  const buf254 = new Float32Array(1048576);
  kernel_37(buf254, buf250, buf248);
  const buf255 = new Float32Array(1048576);
  kernel_38(buf255, buf227);
  const buf257 = new Float32Array(8192);
  kernel_39(buf257, buf254, buf255);
  const buf259 = new Float32Array(8192);
  kernel_40(buf259, buf257);
  const buf260 = new Float32Array(8192);
  kernel_41(buf260, buf259);
  const buf263 = new Float32Array(524288);
  kernel_42(buf263, buf260);
  const buf264 = new Float32Array(524288);
  kernel_43(buf264, buf190);
  const buf266 = new Float32Array(8192);
  kernel_44(buf266, buf263, buf264);
  const buf269 = new Float32Array(8192);
  kernel_45(buf269, buf185, buf266, buf191);
  const buf270 = new Float32Array(128);
  kernel_11(buf270, buf269);
  const buf271 = new Float32Array(1);
  buf271[0] = f32(0.015625);
  const buf274 = new Float32Array(8192);
  kernel_12(buf274, buf269, buf270, buf271);
  const buf276 = new Float32Array(128);
  kernel_13(buf276, buf274);
  const buf277 = new Float32Array(1);
  buf277[0] = f32(0.015625);
  const buf279 = new Float32Array(1);
  buf279[0] = f32(0.00001);
  const buf280 = new Float32Array(128);
  kernel_14(buf280, buf276, buf277, buf279);
  const buf281 = new Float32Array(128);
  kernel_15(buf281, buf280);
  const buf282 = new Float32Array(128);
  kernel_16(buf282, buf281);
  const buf283 = new Float32Array(8192);
  kernel_17(buf283, buf274, buf282);
  const buf288 = new Float32Array(2097152);
  kernel_46(buf288, buf283, buf192, buf193);
  const buf289 = new Float32Array(2097152);
  kernel_47(buf289, buf194);
  const buf291 = new Float32Array(32768);
  kernel_48(buf291, buf288, buf289);
  const buf293 = new Float32Array(32768);
  kernel_49(buf293, buf291, buf195);
  const buf294 = new Float32Array(32768);
  kernel_50(buf294, buf293);
  const buf295 = new Float32Array(32768);
  kernel_51(buf295, buf294, buf293);
  const buf296 = new Float32Array(1);
  buf296[0] = f32(0.7978845608028654);
  const buf297 = new Float32Array(1);
  buf297[0] = f32(0.044715);
  const buf299 = new Float32Array(32768);
  kernel_52(buf299, buf293, buf297, buf295);
  const buf301 = new Float32Array(1);
  buf301[0] = f32(10.0);
  const buf302 = new Float32Array(1);
  kernel_9(buf302, buf301);
  const buf303 = new Float32Array(1);
  buf303[0] = f32(10.0);
  const buf304 = new Float32Array(32768);
  kernel_53(buf304, buf296, buf299);
  const buf305 = new Float32Array(1);
  kernel_9(buf305, buf303);
  const buf307 = new Float32Array(32768);
  kernel_54(buf307, buf304, buf305);
  const buf308 = new Float32Array(32768);
  kernel_55(buf308, buf307, buf302);
  const buf309 = new Float32Array(1);
  buf309[0] = f32(2.0);
  const buf310 = new Float32Array(32768);
  kernel_56(buf310, buf308, buf309);
  const buf311 = new Float32Array(1);
  buf311[0] = f32(1.4426950408889634);
  const buf312 = new Float32Array(32768);
  kernel_56(buf312, buf310, buf311);
  const buf313 = new Float32Array(32768);
  kernel_57(buf313, buf312);
  const buf314 = new Float32Array(1);
  buf314[0] = f32(1.0);
  const buf316 = new Float32Array(32768);
  kernel_58(buf316, buf313, buf314);
  const buf317 = new Float32Array(1);
  buf317[0] = f32(1.0);
  const buf318 = new Float32Array(32768);
  kernel_59(buf318, buf313, buf317);
  const buf319 = new Float32Array(32768);
  kernel_60(buf319, buf318);
  const buf321 = new Float32Array(1);
  buf321[0] = f32(0.5);
  const buf322 = new Float32Array(32768);
  kernel_61(buf322, buf321, buf293);
  const buf323 = new Float32Array(1);
  buf323[0] = f32(1.0);
  const buf324 = new Float32Array(32768);
  kernel_62(buf324, buf323, buf316, buf319);
  const buf328 = new Float32Array(2097152);
  kernel_63(buf328, buf322, buf324);
  const buf329 = new Float32Array(2097152);
  kernel_64(buf329, buf196);
  const buf331 = new Float32Array(8192);
  kernel_65(buf331, buf328, buf329);
  const buf334 = new Float32Array(8192);
  kernel_45(buf334, buf269, buf331, buf197);
  const buf335 = input_27;
  const buf336 = input_28;
  const buf337 = new Float32Array(128);
  kernel_11(buf337, buf334);
  const buf338 = new Float32Array(1);
  buf338[0] = f32(0.015625);
  const buf341 = new Float32Array(8192);
  kernel_12(buf341, buf334, buf337, buf338);
  const buf343 = new Float32Array(128);
  kernel_13(buf343, buf341);
  const buf344 = new Float32Array(1);
  buf344[0] = f32(0.015625);
  const buf346 = new Float32Array(1);
  buf346[0] = f32(0.00001);
  const buf347 = new Float32Array(128);
  kernel_14(buf347, buf343, buf344, buf346);
  const buf348 = new Float32Array(128);
  kernel_15(buf348, buf347);
  const buf349 = new Float32Array(128);
  kernel_16(buf349, buf348);
  const buf350 = new Float32Array(8192);
  kernel_17(buf350, buf341, buf349);
  const buf358 = new Float32Array(811008);
  kernel_66(buf358, buf350, buf335, buf336);
  const buf359 = new Float32Array(811008);
  kernel_67(buf359, buf1);
  const buf361 = new Float32Array(12672);
  kernel_68(buf361, buf358, buf359);
  const buf362 = new Float32Array(12672);
  kernel_69(buf362, buf361);
  const buf363 = targets;
  const buf364 = new Float32Array(99);
  for (let i: i32 = 0; i < 99; i++) buf364[i] = f32(i);
  const buf366 = new Float32Array(12672);
  kernel_0(buf366, buf364);
  const buf368 = new Float32Array(12672);
  kernel_1(buf368, buf363);
  const buf369 = new Float32Array(1);
  buf369[0] = f32(0.5);
  const buf372 = new Float32Array(1);
  buf372[0] = f32(0.5);
  const buf374 = new Float32Array(12672);
  kernel_2(buf374, buf366, buf369, buf368);
  const buf375 = new Float32Array(12672);
  kernel_3(buf375, buf368, buf366, buf372);
  const buf376 = new Float32Array(12672);
  kernel_70(buf376, buf374, buf375);
  const buf377 = new Float32Array(128);
  kernel_71(buf377, buf362);
  const buf379 = new Float32Array(12672);
  kernel_72(buf379, buf362, buf377);
  const buf380 = new Float32Array(1);
  buf380[0] = f32(1.4426950408889634);
  const buf381 = new Float32Array(12672);
  kernel_73(buf381, buf379, buf380);
  const buf383 = new Float32Array(128);
  kernel_74(buf383, buf381);
  const buf384 = new Float32Array(128);
  kernel_75(buf384, buf383);
  const buf385 = new Float32Array(1);
  buf385[0] = f32(0.6931471805599453);
  const buf388 = new Float32Array(12672);
  kernel_76(buf388, buf379, buf384, buf385);
  const buf390 = new Float32Array(128);
  kernel_77(buf390, buf376, buf388);
  const buf391 = new Float32Array(128);
  kernel_78(buf391, buf390);
  const buf392 = new Float32Array(128);
  kernel_78(buf392, buf391);
  const buf393 = new Float32Array(1);
  kernel_79(buf393, buf392);
  const buf394 = new Float32Array(1);
  kernel_9(buf394, buf393);
  const buf395 = new Float32Array(1);
  buf395[0] = f32(0.0078125);
  const buf396 = new Float32Array(1);
  kernel_80(buf396, buf394, buf395);
  const buf397 = new Float32Array(1);
  buf397[0] = f32(1.0);
  const buf402 = new Float32Array(1);
  kernel_81(buf402, buf397, buf395);
  const buf403 = new Float32Array(128);
  kernel_82(buf403, buf402);
  const buf406 = new Float32Array(12672);
  kernel_1(buf406, buf403);
  const buf407 = new Float32Array(12672);
  kernel_70(buf407, buf406, buf388);
  const buf408 = new Float32Array(12672);
  kernel_70(buf408, buf406, buf376);
  const buf409 = new Float32Array(128);
  kernel_83(buf409, buf408);
  const buf410 = new Float32Array(128);
  kernel_84(buf410, buf409);
  const buf413 = new Float32Array(128);
  kernel_85(buf413, buf410, buf384);
  const buf414 = new Float32Array(1);
  kernel_86(buf414, buf413);
  const buf415 = new Float32Array(1);
  kernel_87(buf415, buf414);
  const buf417 = new Float32Array(1);
  buf417[0] = f32(0.6931471805599453);
  const buf423 = new Float32Array(1);
  buf423[0] = f32(0.6931471805599453);
  const buf425 = new Float32Array(12672);
  kernel_88(buf425, buf410, buf385, buf383, buf417, buf381, buf423);
  const buf428 = new Float32Array(12672);
  kernel_89(buf428, buf425, buf379);
  const buf429 = new Float32Array(99);
  kernel_90(buf429, buf428);
  const buf430 = new Float32Array(1);
  kernel_91(buf430, buf429);
  const buf432 = new Float32Array(12672);
  kernel_92(buf432, buf408, buf425, buf380);
  const buf433 = new Float32Array(128);
  kernel_83(buf433, buf432);
  const buf437 = new Float32Array(1);
  buf437[0] = f32(1.0);
  const buf439 = new Float32Array(12672);
  kernel_93(buf439, buf437, buf362, buf377);
  const buf440 = new Float32Array(128);
  kernel_83(buf440, buf439);
  const buf450 = new Float32Array(811008);
  kernel_94(buf450, buf432, buf433, buf439, buf440);
  const buf453 = new Float32Array(6336);
  kernel_95(buf453, buf450, buf358);
  const buf455 = new Float32Array(8192);
  kernel_96(buf455, buf450, buf359);
  const buf457 = new Float32Array(8192);
  kernel_41(buf457, buf455);
  const buf461 = new Float32Array(64);
  kernel_97(buf461, buf457);
  const buf462 = new Float32Array(64);
  kernel_98(buf462, buf461);
  const buf463 = new Float32Array(8192);
  kernel_99(buf463, buf457, buf335);
  const buf465 = new Float32Array(64);
  kernel_100(buf465, buf457, buf350);
  const buf466 = new Float32Array(64);
  kernel_98(buf466, buf465);
  const buf469 = new Float32Array(128);
  kernel_101(buf469, buf463, buf341);
  const buf470 = new Float32Array(128);
  kernel_16(buf470, buf348);
  const buf474 = new Float32Array(1);
  buf474[0] = f32(2.0);
  const buf478 = new Float32Array(128);
  kernel_102(buf478, buf469, buf470, buf474, buf347);
  const buf479 = new Float32Array(128);
  kernel_78(buf479, buf478);
  const buf480 = new Float32Array(1);
  kernel_86(buf480, buf479);
  const buf481 = new Float32Array(1);
  kernel_87(buf481, buf480);
  const buf485 = new Float32Array(128);
  kernel_85(buf485, buf478, buf343);
  const buf486 = new Float32Array(1);
  kernel_86(buf486, buf485);
  const buf487 = new Float32Array(1);
  kernel_87(buf487, buf486);
  const buf489 = new Float32Array(8192);
  kernel_103(buf489, buf478, buf344);
  const buf493 = new Float32Array(8192);
  kernel_104(buf493, buf463, buf349, buf489, buf341);
  const buf494 = new Float32Array(128);
  kernel_11(buf494, buf493);
  const buf495 = new Float32Array(128);
  kernel_84(buf495, buf494);
  const buf498 = new Float32Array(128);
  kernel_85(buf498, buf495, buf337);
  const buf499 = new Float32Array(1);
  kernel_86(buf499, buf498);
  const buf500 = new Float32Array(1);
  kernel_87(buf500, buf499);
  const buf503 = new Float32Array(8192);
  kernel_105(buf503, buf493, buf495, buf338);
  const buf504 = new Float32Array(64);
  kernel_97(buf504, buf503);
  const buf505 = new Float32Array(64);
  kernel_98(buf505, buf504);
  const buf507 = new Float32Array(2097152);
  kernel_106(buf507, buf503);
  const buf510 = new Float32Array(16384);
  kernel_107(buf510, buf507, buf328);
  const buf512 = new Float32Array(32768);
  kernel_108(buf512, buf507, buf329);
  const buf513 = new Float32Array(16384);
  kernel_109(buf513, buf510);
  const buf514 = new Float32Array(32768);
  kernel_110(buf514, buf512);
  const buf515 = new Float32Array(32768);
  kernel_51(buf515, buf514, buf324);
  const buf516 = new Float32Array(32768);
  kernel_51(buf516, buf514, buf322);
  const buf517 = new Float32Array(32768);
  kernel_111(buf517, buf516);
  const buf518 = new Float32Array(256);
  kernel_112(buf518, buf517);
  const buf519 = new Float32Array(1);
  kernel_113(buf519, buf518);
  const buf522 = new Float32Array(32768);
  kernel_114(buf522, buf515, buf293);
  const buf523 = new Float32Array(256);
  kernel_112(buf523, buf522);
  const buf524 = new Float32Array(1);
  kernel_113(buf524, buf523);
  const buf527 = new Float32Array(32768);
  kernel_51(buf527, buf516, buf319);
  const buf529 = new Float32Array(32768);
  kernel_60(buf529, buf318);
  const buf532 = new Float32Array(32768);
  kernel_115(buf532, buf516, buf316, buf529);
  const buf533 = new Float32Array(32768);
  kernel_111(buf533, buf532);
  const buf534 = new Float32Array(256);
  kernel_112(buf534, buf533);
  const buf535 = new Float32Array(1);
  kernel_113(buf535, buf534);
  const buf537 = new Float32Array(32768);
  kernel_111(buf537, buf527);
  const buf538 = new Float32Array(256);
  kernel_112(buf538, buf537);
  const buf539 = new Float32Array(1);
  kernel_113(buf539, buf538);
  const buf544 = new Float32Array(1);
  buf544[0] = f32(0.6931471805599453);
  const buf546 = new Float32Array(32768);
  kernel_116(buf546, buf532, buf527, buf312, buf544);
  const buf547 = new Float32Array(32768);
  kernel_56(buf547, buf546, buf311);
  const buf549 = new Float32Array(32768);
  kernel_114(buf549, buf546, buf310);
  const buf550 = new Float32Array(256);
  kernel_112(buf550, buf549);
  const buf551 = new Float32Array(1);
  kernel_113(buf551, buf550);
  const buf553 = new Float32Array(32768);
  kernel_56(buf553, buf547, buf309);
  const buf555 = new Float32Array(32768);
  kernel_114(buf555, buf547, buf308);
  const buf556 = new Float32Array(256);
  kernel_112(buf556, buf555);
  const buf557 = new Float32Array(1);
  kernel_113(buf557, buf556);
  const buf559 = new Float32Array(32768);
  kernel_117(buf559, buf307, buf302);
  const buf560 = new Float32Array(1);
  buf560[0] = f32(1.0);
  const buf565 = new Float32Array(32768);
  kernel_114(buf565, buf553, buf559);
  const buf566 = new Float32Array(256);
  kernel_112(buf566, buf565);
  const buf567 = new Float32Array(1);
  kernel_113(buf567, buf566);
  const buf569 = new Float32Array(32768);
  kernel_118(buf569, buf553, buf560, buf559);
  const buf570 = new Float32Array(32768);
  kernel_117(buf570, buf304, buf305);
  const buf571 = new Float32Array(1);
  buf571[0] = f32(1.0);
  const buf576 = new Float32Array(32768);
  kernel_114(buf576, buf569, buf570);
  const buf577 = new Float32Array(256);
  kernel_112(buf577, buf576);
  const buf578 = new Float32Array(1);
  kernel_113(buf578, buf577);
  const buf581 = new Float32Array(32768);
  kernel_118(buf581, buf569, buf571, buf570);
  const buf584 = new Float32Array(32768);
  kernel_114(buf584, buf581, buf299);
  const buf585 = new Float32Array(256);
  kernel_112(buf585, buf584);
  const buf586 = new Float32Array(1);
  kernel_113(buf586, buf585);
  const buf588 = new Float32Array(32768);
  kernel_56(buf588, buf581, buf296);
  const buf591 = new Float32Array(32768);
  kernel_114(buf591, buf588, buf295);
  const buf592 = new Float32Array(256);
  kernel_112(buf592, buf591);
  const buf593 = new Float32Array(1);
  kernel_113(buf593, buf592);
  const buf595 = new Float32Array(32768);
  kernel_56(buf595, buf588, buf297);
  const buf596 = new Float32Array(32768);
  kernel_51(buf596, buf595, buf293);
  const buf602 = new Float32Array(32768);
  kernel_119(buf602, buf515, buf321, buf588, buf595, buf294, buf596, buf293);
  const buf603 = new Float32Array(256);
  kernel_112(buf603, buf602);
  const buf604 = new Float32Array(256);
  kernel_120(buf604, buf603);
  const buf606 = new Float32Array(2097152);
  kernel_121(buf606, buf602);
  const buf609 = new Float32Array(16384);
  kernel_122(buf609, buf606, buf288);
  const buf611 = new Float32Array(8192);
  kernel_123(buf611, buf606, buf289);
  const buf612 = new Float32Array(16384);
  kernel_124(buf612, buf609);
  const buf613 = new Float32Array(8192);
  kernel_41(buf613, buf611);
  const buf614 = new Float32Array(64);
  kernel_97(buf614, buf613);
  const buf615 = new Float32Array(64);
  kernel_98(buf615, buf614);
  const buf616 = new Float32Array(8192);
  kernel_99(buf616, buf613, buf192);
  const buf618 = new Float32Array(64);
  kernel_100(buf618, buf613, buf283);
  const buf619 = new Float32Array(64);
  kernel_98(buf619, buf618);
  const buf622 = new Float32Array(128);
  kernel_101(buf622, buf616, buf274);
  const buf623 = new Float32Array(128);
  kernel_16(buf623, buf281);
  const buf627 = new Float32Array(1);
  buf627[0] = f32(2.0);
  const buf631 = new Float32Array(128);
  kernel_102(buf631, buf622, buf623, buf627, buf280);
  const buf632 = new Float32Array(128);
  kernel_78(buf632, buf631);
  const buf633 = new Float32Array(1);
  kernel_86(buf633, buf632);
  const buf634 = new Float32Array(1);
  kernel_87(buf634, buf633);
  const buf638 = new Float32Array(128);
  kernel_85(buf638, buf631, buf276);
  const buf639 = new Float32Array(1);
  kernel_86(buf639, buf638);
  const buf640 = new Float32Array(1);
  kernel_87(buf640, buf639);
  const buf642 = new Float32Array(8192);
  kernel_103(buf642, buf631, buf277);
  const buf646 = new Float32Array(8192);
  kernel_104(buf646, buf616, buf282, buf642, buf274);
  const buf647 = new Float32Array(128);
  kernel_11(buf647, buf646);
  const buf649 = new Float32Array(128);
  kernel_84(buf649, buf647);
  const buf652 = new Float32Array(128);
  kernel_85(buf652, buf649, buf270);
  const buf653 = new Float32Array(1);
  kernel_86(buf653, buf652);
  const buf654 = new Float32Array(1);
  kernel_87(buf654, buf653);
  const buf657 = new Float32Array(8192);
  kernel_125(buf657, buf503, buf646, buf649, buf271);
  const buf658 = new Float32Array(64);
  kernel_97(buf658, buf657);
  const buf659 = new Float32Array(64);
  kernel_98(buf659, buf658);
  const buf661 = new Float32Array(524288);
  kernel_126(buf661, buf657);
  const buf664 = new Float32Array(4096);
  kernel_127(buf664, buf661, buf263);
  const buf666 = new Float32Array(8192);
  kernel_128(buf666, buf661, buf264);
  const buf667 = new Float32Array(4096);
  kernel_129(buf667, buf664);
  const buf668 = new Float32Array(8192);
  kernel_41(buf668, buf666);
  const buf669 = new Float32Array(8192);
  kernel_130(buf669, buf668);
  const buf672 = new Float32Array(1048576);
  kernel_131(buf672, buf669);
  const buf675 = new Float32Array(8192);
  kernel_132(buf675, buf672, buf254);
  const buf676 = new Float32Array(65536);
  kernel_133(buf676, buf672, buf255);
  const buf678 = new Float32Array(65536);
  kernel_29(buf678, buf676);
  const buf680 = new Float32Array(512);
  kernel_134(buf680, buf678, buf248);
  const buf682 = new Float32Array(512);
  kernel_36(buf682, buf249);
  const buf689 = new Float32Array(1);
  buf689[0] = f32(0.6931471805599453);
  const buf691 = new Float32Array(65536);
  kernel_135(buf691, buf678, buf250, buf680, buf682, buf247, buf689);
  const buf692 = new Float32Array(65536);
  kernel_33(buf692, buf691, buf246);
  const buf694 = new Float32Array(65536);
  kernel_136(buf694, buf691, buf245);
  const buf695 = new Float32Array(16384);
  kernel_137(buf695, buf694);
  const buf696 = new Float32Array(128);
  kernel_138(buf696, buf695);
  const buf697 = new Float32Array(1);
  kernel_139(buf697, buf696);
  const buf699 = new Float32Array(512);
  kernel_35(buf699, buf692);
  const buf703 = new Float32Array(1);
  buf703[0] = f32(1.0);
  const buf705 = new Float32Array(65536);
  kernel_140(buf705, buf703, buf242, buf243);
  const buf706 = new Float32Array(512);
  kernel_35(buf706, buf705);
  const buf712 = new Float32Array(65536);
  kernel_141(buf712, buf692, buf699, buf705, buf706);
  const buf715 = new Float32Array(65536);
  kernel_136(buf715, buf712, buf239);
  const buf716 = new Float32Array(16384);
  kernel_137(buf716, buf715);
  const buf717 = new Float32Array(128);
  kernel_138(buf717, buf716);
  const buf718 = new Float32Array(1);
  kernel_139(buf718, buf717);
  const buf721 = new Float32Array(1048576);
  kernel_142(buf721, buf712, buf240);
  const buf724 = new Float32Array(8192);
  kernel_143(buf724, buf721, buf235);
  const buf725 = new Float32Array(8192);
  kernel_144(buf725, buf721, buf236);
  const buf732 = new Float32Array(8192);
  kernel_145(buf732, buf675);
  const buf733 = new Float32Array(24576);
  for (let i: i32 = 0; i < 24576; i++) buf733[i] = f32(0.0);
  for (let ai: i32 = 0; ai < 8192; ai++) {
    const d0: i32 = (ai / 8192) % 1;
    const d1: i32 = (ai / 64) % 128;
    const d2: i32 = (ai / 64) % 1;
    const d3: i32 = (ai / 16) % 4;
    const d4: i32 = ai % 16;
    buf733[(d0 + 0) * 24576 + (d1 + 0) * 192 + (d2 + 2) * 64 + (d3 + 0) * 16 + (d4 + 0) * 1] = buf732[ai];
  }
  const buf734 = new Float32Array(8192);
  kernel_146(buf734, buf724);
  const buf735 = new Float32Array(24576);
  for (let i: i32 = 0; i < 24576; i++) buf735[i] = f32(0.0);
  for (let ai: i32 = 0; ai < 8192; ai++) {
    const d0: i32 = (ai / 8192) % 1;
    const d1: i32 = (ai / 64) % 128;
    const d2: i32 = (ai / 64) % 1;
    const d3: i32 = (ai / 16) % 4;
    const d4: i32 = ai % 16;
    buf735[(d0 + 0) * 24576 + (d1 + 0) * 192 + (d2 + 1) * 64 + (d3 + 0) * 16 + (d4 + 0) * 1] = buf734[ai];
  }
  const buf737 = new Float32Array(8192);
  kernel_145(buf737, buf725);
  const buf738 = new Float32Array(24576);
  for (let i: i32 = 0; i < 24576; i++) buf738[i] = f32(0.0);
  for (let ai: i32 = 0; ai < 8192; ai++) {
    const d0: i32 = (ai / 8192) % 1;
    const d1: i32 = (ai / 64) % 128;
    const d2: i32 = (ai / 64) % 1;
    const d3: i32 = (ai / 16) % 4;
    const d4: i32 = ai % 16;
    buf738[(d0 + 0) * 24576 + (d1 + 0) * 192 + (d2 + 0) * 64 + (d3 + 0) * 16 + (d4 + 0) * 1] = buf737[ai];
  }
  const buf739 = new Float32Array(24576);
  kernel_147(buf739, buf733, buf735, buf738);
  const buf740 = new Float32Array(24576);
  kernel_148(buf740, buf739);
  const buf741 = new Float32Array(192);
  kernel_149(buf741, buf740);
  const buf742 = new Float32Array(192);
  kernel_150(buf742, buf741);
  const buf744 = new Float32Array(1572864);
  kernel_151(buf744, buf740);
  const buf747 = new Float32Array(12288);
  kernel_152(buf747, buf744, buf216);
  const buf749 = new Float32Array(8192);
  kernel_153(buf749, buf744, buf217);
  const buf750 = new Float32Array(12288);
  kernel_154(buf750, buf747);
  const buf751 = new Float32Array(8192);
  kernel_41(buf751, buf749);
  const buf752 = new Float32Array(64);
  kernel_97(buf752, buf751);
  const buf753 = new Float32Array(64);
  kernel_98(buf753, buf752);
  const buf754 = new Float32Array(8192);
  kernel_99(buf754, buf751, buf186);
  const buf756 = new Float32Array(64);
  kernel_100(buf756, buf751, buf211);
  const buf757 = new Float32Array(64);
  kernel_98(buf757, buf756);
  const buf760 = new Float32Array(128);
  kernel_101(buf760, buf754, buf202);
  const buf761 = new Float32Array(128);
  kernel_16(buf761, buf209);
  const buf765 = new Float32Array(1);
  buf765[0] = f32(2.0);
  const buf769 = new Float32Array(128);
  kernel_102(buf769, buf760, buf761, buf765, buf208);
  const buf770 = new Float32Array(128);
  kernel_78(buf770, buf769);
  const buf771 = new Float32Array(1);
  kernel_86(buf771, buf770);
  const buf772 = new Float32Array(1);
  kernel_87(buf772, buf771);
  const buf776 = new Float32Array(128);
  kernel_85(buf776, buf769, buf204);
  const buf777 = new Float32Array(1);
  kernel_86(buf777, buf776);
  const buf778 = new Float32Array(1);
  kernel_87(buf778, buf777);
  const buf780 = new Float32Array(8192);
  kernel_103(buf780, buf769, buf205);
  const buf784 = new Float32Array(8192);
  kernel_104(buf784, buf754, buf210, buf780, buf202);
  const buf785 = new Float32Array(128);
  kernel_11(buf785, buf784);
  const buf787 = new Float32Array(128);
  kernel_84(buf787, buf785);
  const buf790 = new Float32Array(128);
  kernel_85(buf790, buf787, buf198);
  const buf791 = new Float32Array(1);
  kernel_86(buf791, buf790);
  const buf792 = new Float32Array(1);
  kernel_87(buf792, buf791);
  const buf795 = new Float32Array(8192);
  kernel_125(buf795, buf657, buf784, buf787, buf199);
  const buf796 = new Float32Array(64);
  kernel_97(buf796, buf795);
  const buf797 = new Float32Array(64);
  kernel_98(buf797, buf796);
  const buf799 = new Float32Array(2097152);
  kernel_106(buf799, buf795);
  const buf802 = new Float32Array(16384);
  kernel_107(buf802, buf799, buf179);
  const buf804 = new Float32Array(32768);
  kernel_108(buf804, buf799, buf180);
  const buf805 = new Float32Array(16384);
  kernel_109(buf805, buf802);
  const buf806 = new Float32Array(32768);
  kernel_110(buf806, buf804);
  const buf807 = new Float32Array(32768);
  kernel_51(buf807, buf806, buf175);
  const buf808 = new Float32Array(32768);
  kernel_51(buf808, buf806, buf173);
  const buf809 = new Float32Array(32768);
  kernel_111(buf809, buf808);
  const buf810 = new Float32Array(256);
  kernel_112(buf810, buf809);
  const buf811 = new Float32Array(1);
  kernel_113(buf811, buf810);
  const buf814 = new Float32Array(32768);
  kernel_114(buf814, buf807, buf144);
  const buf815 = new Float32Array(256);
  kernel_112(buf815, buf814);
  const buf816 = new Float32Array(1);
  kernel_113(buf816, buf815);
  const buf819 = new Float32Array(32768);
  kernel_51(buf819, buf808, buf170);
  const buf821 = new Float32Array(32768);
  kernel_60(buf821, buf169);
  const buf824 = new Float32Array(32768);
  kernel_115(buf824, buf808, buf167, buf821);
  const buf825 = new Float32Array(32768);
  kernel_111(buf825, buf824);
  const buf826 = new Float32Array(256);
  kernel_112(buf826, buf825);
  const buf827 = new Float32Array(1);
  kernel_113(buf827, buf826);
  const buf829 = new Float32Array(32768);
  kernel_111(buf829, buf819);
  const buf830 = new Float32Array(256);
  kernel_112(buf830, buf829);
  const buf831 = new Float32Array(1);
  kernel_113(buf831, buf830);
  const buf836 = new Float32Array(1);
  buf836[0] = f32(0.6931471805599453);
  const buf838 = new Float32Array(32768);
  kernel_116(buf838, buf824, buf819, buf163, buf836);
  const buf839 = new Float32Array(32768);
  kernel_56(buf839, buf838, buf162);
  const buf841 = new Float32Array(32768);
  kernel_114(buf841, buf838, buf161);
  const buf842 = new Float32Array(256);
  kernel_112(buf842, buf841);
  const buf843 = new Float32Array(1);
  kernel_113(buf843, buf842);
  const buf845 = new Float32Array(32768);
  kernel_56(buf845, buf839, buf160);
  const buf847 = new Float32Array(32768);
  kernel_114(buf847, buf839, buf159);
  const buf848 = new Float32Array(256);
  kernel_112(buf848, buf847);
  const buf849 = new Float32Array(1);
  kernel_113(buf849, buf848);
  const buf851 = new Float32Array(32768);
  kernel_117(buf851, buf158, buf153);
  const buf852 = new Float32Array(1);
  buf852[0] = f32(1.0);
  const buf857 = new Float32Array(32768);
  kernel_114(buf857, buf845, buf851);
  const buf858 = new Float32Array(256);
  kernel_112(buf858, buf857);
  const buf859 = new Float32Array(1);
  kernel_113(buf859, buf858);
  const buf861 = new Float32Array(32768);
  kernel_118(buf861, buf845, buf852, buf851);
  const buf862 = new Float32Array(32768);
  kernel_117(buf862, buf155, buf156);
  const buf863 = new Float32Array(1);
  buf863[0] = f32(1.0);
  const buf868 = new Float32Array(32768);
  kernel_114(buf868, buf861, buf862);
  const buf869 = new Float32Array(256);
  kernel_112(buf869, buf868);
  const buf870 = new Float32Array(1);
  kernel_113(buf870, buf869);
  const buf873 = new Float32Array(32768);
  kernel_118(buf873, buf861, buf863, buf862);
  const buf876 = new Float32Array(32768);
  kernel_114(buf876, buf873, buf150);
  const buf877 = new Float32Array(256);
  kernel_112(buf877, buf876);
  const buf878 = new Float32Array(1);
  kernel_113(buf878, buf877);
  const buf880 = new Float32Array(32768);
  kernel_56(buf880, buf873, buf147);
  const buf883 = new Float32Array(32768);
  kernel_114(buf883, buf880, buf146);
  const buf884 = new Float32Array(256);
  kernel_112(buf884, buf883);
  const buf885 = new Float32Array(1);
  kernel_113(buf885, buf884);
  const buf887 = new Float32Array(32768);
  kernel_56(buf887, buf880, buf148);
  const buf888 = new Float32Array(32768);
  kernel_51(buf888, buf887, buf144);
  const buf894 = new Float32Array(32768);
  kernel_119(buf894, buf807, buf172, buf880, buf887, buf145, buf888, buf144);
  const buf895 = new Float32Array(256);
  kernel_112(buf895, buf894);
  const buf896 = new Float32Array(256);
  kernel_120(buf896, buf895);
  const buf898 = new Float32Array(2097152);
  kernel_121(buf898, buf894);
  const buf901 = new Float32Array(16384);
  kernel_122(buf901, buf898, buf139);
  const buf903 = new Float32Array(8192);
  kernel_123(buf903, buf898, buf140);
  const buf904 = new Float32Array(16384);
  kernel_124(buf904, buf901);
  const buf905 = new Float32Array(8192);
  kernel_41(buf905, buf903);
  const buf906 = new Float32Array(64);
  kernel_97(buf906, buf905);
  const buf907 = new Float32Array(64);
  kernel_98(buf907, buf906);
  const buf908 = new Float32Array(8192);
  kernel_99(buf908, buf905, buf43);
  const buf910 = new Float32Array(64);
  kernel_100(buf910, buf905, buf134);
  const buf911 = new Float32Array(64);
  kernel_98(buf911, buf910);
  const buf914 = new Float32Array(128);
  kernel_101(buf914, buf908, buf125);
  const buf915 = new Float32Array(128);
  kernel_16(buf915, buf132);
  const buf919 = new Float32Array(1);
  buf919[0] = f32(2.0);
  const buf923 = new Float32Array(128);
  kernel_102(buf923, buf914, buf915, buf919, buf131);
  const buf924 = new Float32Array(128);
  kernel_78(buf924, buf923);
  const buf925 = new Float32Array(1);
  kernel_86(buf925, buf924);
  const buf926 = new Float32Array(1);
  kernel_87(buf926, buf925);
  const buf930 = new Float32Array(128);
  kernel_85(buf930, buf923, buf127);
  const buf931 = new Float32Array(1);
  kernel_86(buf931, buf930);
  const buf932 = new Float32Array(1);
  kernel_87(buf932, buf931);
  const buf934 = new Float32Array(8192);
  kernel_103(buf934, buf923, buf128);
  const buf938 = new Float32Array(8192);
  kernel_104(buf938, buf908, buf133, buf934, buf125);
  const buf939 = new Float32Array(128);
  kernel_11(buf939, buf938);
  const buf941 = new Float32Array(128);
  kernel_84(buf941, buf939);
  const buf944 = new Float32Array(128);
  kernel_85(buf944, buf941, buf121);
  const buf945 = new Float32Array(1);
  kernel_86(buf945, buf944);
  const buf946 = new Float32Array(1);
  kernel_87(buf946, buf945);
  const buf949 = new Float32Array(8192);
  kernel_125(buf949, buf795, buf938, buf941, buf122);
  const buf950 = new Float32Array(64);
  kernel_97(buf950, buf949);
  const buf951 = new Float32Array(64);
  kernel_98(buf951, buf950);
  const buf953 = new Float32Array(524288);
  kernel_126(buf953, buf949);
  const buf956 = new Float32Array(4096);
  kernel_127(buf956, buf953, buf114);
  const buf958 = new Float32Array(8192);
  kernel_128(buf958, buf953, buf115);
  const buf959 = new Float32Array(4096);
  kernel_129(buf959, buf956);
  const buf960 = new Float32Array(8192);
  kernel_41(buf960, buf958);
  const buf961 = new Float32Array(8192);
  kernel_130(buf961, buf960);
  const buf964 = new Float32Array(1048576);
  kernel_131(buf964, buf961);
  const buf967 = new Float32Array(8192);
  kernel_132(buf967, buf964, buf105);
  const buf968 = new Float32Array(65536);
  kernel_133(buf968, buf964, buf106);
  const buf970 = new Float32Array(65536);
  kernel_29(buf970, buf968);
  const buf972 = new Float32Array(512);
  kernel_134(buf972, buf970, buf99);
  const buf974 = new Float32Array(512);
  kernel_36(buf974, buf100);
  const buf981 = new Float32Array(1);
  buf981[0] = f32(0.6931471805599453);
  const buf983 = new Float32Array(65536);
  kernel_135(buf983, buf970, buf101, buf972, buf974, buf98, buf981);
  const buf984 = new Float32Array(65536);
  kernel_33(buf984, buf983, buf97);
  const buf986 = new Float32Array(65536);
  kernel_136(buf986, buf983, buf96);
  const buf987 = new Float32Array(16384);
  kernel_137(buf987, buf986);
  const buf988 = new Float32Array(128);
  kernel_138(buf988, buf987);
  const buf989 = new Float32Array(1);
  kernel_139(buf989, buf988);
  const buf991 = new Float32Array(512);
  kernel_35(buf991, buf984);
  const buf995 = new Float32Array(1);
  buf995[0] = f32(1.0);
  const buf997 = new Float32Array(65536);
  kernel_140(buf997, buf995, buf93, buf94);
  const buf998 = new Float32Array(512);
  kernel_35(buf998, buf997);
  const buf1004 = new Float32Array(65536);
  kernel_141(buf1004, buf984, buf991, buf997, buf998);
  const buf1008 = new Float32Array(65536);
  kernel_136(buf1008, buf1004, buf90);
  const buf1009 = new Float32Array(16384);
  kernel_137(buf1009, buf1008);
  const buf1010 = new Float32Array(128);
  kernel_138(buf1010, buf1009);
  const buf1011 = new Float32Array(1);
  kernel_139(buf1011, buf1010);
  const buf1014 = new Float32Array(1048576);
  kernel_142(buf1014, buf1004, buf91);
  const buf1017 = new Float32Array(8192);
  kernel_143(buf1017, buf1014, buf86);
  const buf1018 = new Float32Array(8192);
  kernel_144(buf1018, buf1014, buf87);
  const buf1025 = new Float32Array(8192);
  kernel_145(buf1025, buf967);
  const buf1026 = new Float32Array(24576);
  for (let i: i32 = 0; i < 24576; i++) buf1026[i] = f32(0.0);
  for (let ai: i32 = 0; ai < 8192; ai++) {
    const d0: i32 = (ai / 8192) % 1;
    const d1: i32 = (ai / 64) % 128;
    const d2: i32 = (ai / 64) % 1;
    const d3: i32 = (ai / 16) % 4;
    const d4: i32 = ai % 16;
    buf1026[(d0 + 0) * 24576 + (d1 + 0) * 192 + (d2 + 2) * 64 + (d3 + 0) * 16 + (d4 + 0) * 1] = buf1025[ai];
  }
  const buf1027 = new Float32Array(8192);
  kernel_146(buf1027, buf1017);
  const buf1028 = new Float32Array(24576);
  for (let i: i32 = 0; i < 24576; i++) buf1028[i] = f32(0.0);
  for (let ai: i32 = 0; ai < 8192; ai++) {
    const d0: i32 = (ai / 8192) % 1;
    const d1: i32 = (ai / 64) % 128;
    const d2: i32 = (ai / 64) % 1;
    const d3: i32 = (ai / 16) % 4;
    const d4: i32 = ai % 16;
    buf1028[(d0 + 0) * 24576 + (d1 + 0) * 192 + (d2 + 1) * 64 + (d3 + 0) * 16 + (d4 + 0) * 1] = buf1027[ai];
  }
  const buf1030 = new Float32Array(8192);
  kernel_145(buf1030, buf1018);
  const buf1031 = new Float32Array(24576);
  for (let i: i32 = 0; i < 24576; i++) buf1031[i] = f32(0.0);
  for (let ai: i32 = 0; ai < 8192; ai++) {
    const d0: i32 = (ai / 8192) % 1;
    const d1: i32 = (ai / 64) % 128;
    const d2: i32 = (ai / 64) % 1;
    const d3: i32 = (ai / 16) % 4;
    const d4: i32 = ai % 16;
    buf1031[(d0 + 0) * 24576 + (d1 + 0) * 192 + (d2 + 0) * 64 + (d3 + 0) * 16 + (d4 + 0) * 1] = buf1030[ai];
  }
  const buf1032 = new Float32Array(24576);
  kernel_147(buf1032, buf1026, buf1028, buf1031);
  const buf1033 = new Float32Array(24576);
  kernel_148(buf1033, buf1032);
  const buf1034 = new Float32Array(192);
  kernel_149(buf1034, buf1033);
  const buf1035 = new Float32Array(192);
  kernel_150(buf1035, buf1034);
  const buf1037 = new Float32Array(1572864);
  kernel_151(buf1037, buf1033);
  const buf1040 = new Float32Array(12288);
  kernel_152(buf1040, buf1037, buf67);
  const buf1042 = new Float32Array(8192);
  kernel_153(buf1042, buf1037, buf68);
  const buf1043 = new Float32Array(12288);
  kernel_154(buf1043, buf1040);
  const buf1044 = new Float32Array(8192);
  kernel_41(buf1044, buf1042);
  const buf1045 = new Float32Array(64);
  kernel_97(buf1045, buf1044);
  const buf1046 = new Float32Array(64);
  kernel_98(buf1046, buf1045);
  const buf1047 = new Float32Array(8192);
  kernel_99(buf1047, buf1044, buf37);
  const buf1049 = new Float32Array(64);
  kernel_100(buf1049, buf1044, buf62);
  const buf1050 = new Float32Array(64);
  kernel_98(buf1050, buf1049);
  const buf1053 = new Float32Array(128);
  kernel_101(buf1053, buf1047, buf53);
  const buf1054 = new Float32Array(128);
  kernel_16(buf1054, buf60);
  const buf1058 = new Float32Array(1);
  buf1058[0] = f32(2.0);
  const buf1062 = new Float32Array(128);
  kernel_102(buf1062, buf1053, buf1054, buf1058, buf59);
  const buf1063 = new Float32Array(128);
  kernel_78(buf1063, buf1062);
  const buf1064 = new Float32Array(1);
  kernel_86(buf1064, buf1063);
  const buf1065 = new Float32Array(1);
  kernel_87(buf1065, buf1064);
  const buf1069 = new Float32Array(128);
  kernel_85(buf1069, buf1062, buf55);
  const buf1070 = new Float32Array(1);
  kernel_86(buf1070, buf1069);
  const buf1071 = new Float32Array(1);
  kernel_87(buf1071, buf1070);
  const buf1073 = new Float32Array(8192);
  kernel_103(buf1073, buf1062, buf56);
  const buf1077 = new Float32Array(8192);
  kernel_104(buf1077, buf1047, buf61, buf1073, buf53);
  const buf1078 = new Float32Array(128);
  kernel_11(buf1078, buf1077);
  const buf1080 = new Float32Array(128);
  kernel_84(buf1080, buf1078);
  const buf1083 = new Float32Array(128);
  kernel_85(buf1083, buf1080, buf49);
  const buf1084 = new Float32Array(1);
  kernel_86(buf1084, buf1083);
  const buf1085 = new Float32Array(1);
  kernel_87(buf1085, buf1084);
  const buf1088 = new Float32Array(8192);
  kernel_125(buf1088, buf949, buf1077, buf1080, buf50);
  const buf1089 = new Float32Array(16384);
  kernel_155(buf1089, buf712, buf1004);
  const buf1092 = new Float32Array(16384);
  kernel_156(buf1092, buf1089, buf32);
  const buf1093 = new Float32Array(16384);
  kernel_157(buf1093, buf1092);
  const buf1094 = new Float32Array(128);
  kernel_138(buf1094, buf1093);
  const buf1095 = new Float32Array(1);
  kernel_139(buf1095, buf1094);
  const buf1098 = new Float32Array(8192);
  kernel_158(buf1098, buf1088);
  const buf1100 = new Float32Array(811008);
  kernel_159(buf1100, buf1088);
  const buf1103 = new Float32Array(6336);
  kernel_160(buf1103, buf1100, buf18);
  const buf1105 = new Float32Array(12672);
  kernel_161(buf1105, buf1100, buf19);
  const buf1107 = new Float32Array(6336);
  kernel_162(buf1107, buf453, buf1103);
  const buf1108 = new Float32Array(12672);
  kernel_69(buf1108, buf1105);
  const buf1110 = new Float32Array(12672);
  kernel_70(buf1110, buf1108, buf13);

  const __out = new Float32Array(1 + 6336 + 8192 + 64 + 64 + 12288 + 192 + 4096 + 64 + 64 + 64 + 16384 + 256 + 16384 + 64 + 64 + 64 + 12288 + 192 + 4096 + 64 + 64 + 64 + 16384 + 256 + 16384 + 64 + 64 + 64);
  let __off: i32 = 0;
  for (let __i: i32 = 0; __i < 1; __i++) __out[__off + __i] = buf396[__i];
  __off += 1;
  for (let __i: i32 = 0; __i < 6336; __i++) __out[__off + __i] = buf1107[__i];
  __off += 6336;
  for (let __i: i32 = 0; __i < 8192; __i++) __out[__off + __i] = buf1098[__i];
  __off += 8192;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf1050[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf1046[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 12288; __i++) __out[__off + __i] = buf1043[__i];
  __off += 12288;
  for (let __i: i32 = 0; __i < 192; __i++) __out[__off + __i] = buf1035[__i];
  __off += 192;
  for (let __i: i32 = 0; __i < 4096; __i++) __out[__off + __i] = buf959[__i];
  __off += 4096;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf951[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf911[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf907[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 16384; __i++) __out[__off + __i] = buf904[__i];
  __off += 16384;
  for (let __i: i32 = 0; __i < 256; __i++) __out[__off + __i] = buf896[__i];
  __off += 256;
  for (let __i: i32 = 0; __i < 16384; __i++) __out[__off + __i] = buf805[__i];
  __off += 16384;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf797[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf757[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf753[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 12288; __i++) __out[__off + __i] = buf750[__i];
  __off += 12288;
  for (let __i: i32 = 0; __i < 192; __i++) __out[__off + __i] = buf742[__i];
  __off += 192;
  for (let __i: i32 = 0; __i < 4096; __i++) __out[__off + __i] = buf667[__i];
  __off += 4096;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf659[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf619[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf615[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 16384; __i++) __out[__off + __i] = buf612[__i];
  __off += 16384;
  for (let __i: i32 = 0; __i < 256; __i++) __out[__off + __i] = buf604[__i];
  __off += 256;
  for (let __i: i32 = 0; __i < 16384; __i++) __out[__off + __i] = buf513[__i];
  __off += 16384;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf505[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf466[__i];
  __off += 64;
  for (let __i: i32 = 0; __i < 64; __i++) __out[__off + __i] = buf462[__i];
  __off += 64;
  return __out;
}
