// *Author*: Podtsepko Igor (@IPodtsepko)
use image::ImageFormat;

/// Maximum recursion depth, when calculating the color of the ray. When the specified value is
/// reached, it is assumed that the ray has a background color.
pub const MAX_DEPTH: usize = 5;

/// Default image width (in pixels).
pub const WIDTH: u32 = 1024;
/// Default image height (in pixels).
pub const HEIGHT: u32 = 768;

/// Default output file name.
pub static FILENAME: &str = "image.png";
/// Default output image format.
pub const IMAGE_FORMAT: ImageFormat = ImageFormat::Png;
