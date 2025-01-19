#![allow(dead_code)]

// *Author*: Podtsepko Igor (@IPodtsepko)
use crate::colors;
use crate::general::Material;

/// Polished ivory.
pub const IVORY: Material = Material::new(1.0, colors::IVORY, [0.6, 0.3, 0.1, 0.0], 50.0);

/// Red-brown rubber.
pub const RED_RUBBER: Material = Material::new(1.0, colors::RED_BROWN, [0.9, 0.1, 0.0, 0.0], 10.0);

/// Mirror material. It has no color itself, but reflects all the rays from itself.
pub const MIRROR: Material = Material::new(1.0, colors::WHITE, [0.0, 10.0, 0.8, 0.0], 1425.0);

/// Glass. Refracts the rays passing through it.
pub const GLASS: Material = Material::new(1.5, colors::LEAFY_GREEN, [0.1, 0.5, 0.1, 0.8], 125.0);

/// A simple approximation of green plastic.
pub const GREEN_PLASTIC: Material = Material::new(1.0, colors::MYRTLE, [1.0, 0.0, 0.0, 0.0], 25.0);
