//! CPython's `random.Random(str)` and `shuffle`, so permutation rows match the reference.
//!
//! The reference shuffles option order with `random.Random(f"{qid}:{n}")`. That is MT19937
//! seeded through `init_by_array` with the integer `int.from_bytes(seed + sha512(seed))`, and
//! `shuffle` draws with `_randbelow_with_getrandbits`.

use sha2::{Digest, Sha512};

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for a string seed (version 2 seeding).
    pub fn from_str_seed(seed: &str) -> Self {
        let mut bytes = seed.as_bytes().to_vec();
        bytes.extend_from_slice(&Sha512::digest(seed.as_bytes()));
        // The seed is that big-endian integer, handed to init_by_array as 32-bit words, least
        // significant first, with no leading zero words.
        let first = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
        let significant = &bytes[first..];
        let mut key: Vec<u32> =
            significant.rchunks(4).map(|chunk| chunk.iter().fold(0u32, |acc, &b| (acc << 8) | b as u32)).collect();
        if key.is_empty() {
            key.push(0);
        }
        let mut rng = PyRandom { mt: [0; N], index: N };
        rng.init_by_array(&key);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            for k in 0..N {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % N] & 0x7fff_ffff);
                let mut value = self.mt[(k + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    value ^= 0x9908_b0df;
                }
                self.mt[k] = value;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `getrandbits(k)` for k <= 32.
    fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 { 0 } else { self.next_u32() >> (32 - k) }
    }

    /// `_randbelow_with_getrandbits(n)`.
    fn randbelow(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `random.shuffle(items)`.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.randbelow(i + 1);
            items.swap(i, j);
        }
    }
}
