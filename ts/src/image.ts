import { open } from "node:fs/promises";
import { createHash } from "node:crypto";
import type { Content } from "./types.js";

const limit = 20 * 1024 * 1024;
const images = new WeakMap<ImageInput, { bytes: Buffer; context: Content; identity: string }>();

/** An immutable snapshot of a JPEG/PNG and optional context. No model or GPU resources. */
export class ImageInput {
  private constructor(bytes: Uint8Array, context: Content = null) {
    if (!bytes.byteLength || bytes.byteLength > limit) throw new RangeError("image must contain 1 byte to 20 MiB");
    const copy = Buffer.from(bytes);
    const text = JSON.stringify(context);
    images.set(this, {
      bytes: copy, context: JSON.parse(text) as Content,
      identity: createHash("sha256").update(copy).digest("hex") + ":" + text,
    });
    Object.freeze(this);
  }

  static fromBytes(bytes: Uint8Array): ImageInput { return new ImageInput(bytes); }

  static async fromFile(path: string): Promise<ImageInput> {
    const file = await open(path, "r");
    try {
      const stat = await file.stat();
      if (!stat.isFile() || stat.size === 0 || stat.size > limit) throw new RangeError("image file must contain 1 byte to 20 MiB");
      // Bounded read even if another process grows the file after stat().
      const bytes = Buffer.alloc(stat.size + 1);
      let used = 0;
      while (used < bytes.length) {
        const { bytesRead } = await file.read(bytes, used, bytes.length - used, null);
        if (!bytesRead) break;
        used += bytesRead;
      }
      if (used > stat.size) throw new RangeError("image file changed while being read");
      return new ImageInput(bytes.subarray(0, used));
    } finally { await file.close(); }
  }

  withContext(context: Content): ImageInput { return new ImageInput(imageData(this).bytes, context); }

  /** Prevent accidental image serialization as ordinary JSON state or calibration data. */
  toJSON(): never { throw new TypeError("ImageInput must be passed directly as the predicate state"); }
}

/** Internal attachment access. Kept out of the package's public exports. */
export function imageData(image: ImageInput) {
  const data = images.get(image);
  if (!data) throw new TypeError("invalid ImageInput");
  return data;
}
