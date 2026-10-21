//! cargo run -p moedex-lars --example image_answers -- /path/to/vision-pack photo.png
use moedex_lars::{ImageInput, Lars, Options, Result};

#[derive(Clone, Copy, Debug, PartialEq, Options)]
enum Color {
    /// Red
    Red,
    /// Blue
    Blue,
    /// Another color
    Other,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: image_answers VISION_PACK IMAGE");
        std::process::exit(2);
    }
    let lars = Lars::open(&args[1]).wait()?;
    let image = ImageInput::from_file(&args[2])?;
    let red = lars.noul("red", "The image is predominantly red");
    let color = lars.choice::<Color>("color", "What is the predominant color in the image?");
    let (red, color) = lars.ask_image(&image, (&red, &color)).wait()?;
    println!("red: {red:?}\ncolor: {color:?}");
    Ok(())
}
