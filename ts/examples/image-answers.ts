// npx tsx examples/image-answers.ts /path/to/vision-pack /path/to/photo.png
import { Lars, type ImageInput } from "../src/index.js";

const [pack, path] = process.argv.slice(2);
if (!pack || !path) throw new Error("usage: image-answers VISION_PACK IMAGE");
const lars = await Lars.open({ pack });
try {
  const image = await Lars.imageFile(path);
  const red = lars.noul<ImageInput>("red", "The image is predominantly red");
  const color = lars.choice("color", "What is the predominant color in the image?", {
    red: "Red", blue: "Blue", other: "Another color",
  }, { state: (image: ImageInput) => image });
  console.log(await lars.ask(image, { red, color }));
} finally { lars.close(); }
