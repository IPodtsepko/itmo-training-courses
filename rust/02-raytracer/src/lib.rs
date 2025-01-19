// *Author*: Podtsepko Igor (@IPodtsepko)

/// A module containing a set of predefined colors that can be used when creating shapes.
pub mod colors;

/// A module with constants for rendering by default.
pub mod defaults;

/// A module containing the main tools for creating images: scene (`Scene`),
/// lighting (`Light`), etc.
pub mod general;

/// A module containing the basic definitions of analytic geometry. First of all - vectors in
/// three-dimensional space (`Vec3`).
pub mod geometry;

/// A module containing a set of predefined light intensity values to simplify the construction
/// of scenes.
pub mod intensity;

/// A module containing a set of predefined materials that can be used when creating shapes.
pub mod materials;

/// A module that provides tools for working with shapes. The main components of the module are the
/// shape interface (`trait Shape`) and the shape factory (`ShapeFactory`).
pub mod shapes;

mod utils;

#[cfg(test)]
mod tests {
    use super::geometry::Vec3;

    #[test]
    fn test_vec3_add() {
        let lhs = Vec3::new(1.0, 5.0, 7.0);
        let rhs = Vec3::new(-5.0, 10.0, 0.0);
        assert_eq!(Vec3::new(-4.0, 15.0, 7.0), lhs + rhs);
    }

    #[test]
    fn test_vec3_sub() {
        let lhs = Vec3::new(1.0, 5.0, 7.0);
        let rhs = Vec3::new(-5.0, 10.0, 0.0);
        assert_eq!(Vec3::new(6.0, -5.0, 7.0), lhs - rhs);
    }

    #[test]
    fn test_vec3_mul() {
        let lhs = Vec3::new(1.0, 5.0, 7.0);
        let rhs = Vec3::new(-5.0, 10.0, 0.0);
        assert_eq!(45.0, lhs * rhs);
        assert_eq!(Vec3::new(2.0, 10.0, 14.0), lhs * 2);
        assert_eq!(Vec3::new(5.0, -10.0, 0.0), -1 * rhs);
    }

    #[test]
    fn test_vec3_neg() {
        let vec = Vec3::new(1.0, 5.0, 7.0);
        assert_eq!(Vec3::new(-1.0, -5.0, -7.0), -vec);
    }

    #[test]
    fn test_vec3_div() {
        let vec = Vec3::new(1.0, 5.0, 7.0);
        assert_eq!(Vec3::new(-0.5, -2.5, -3.5), vec / -2);
    }

    #[test]
    fn test_vec3_len() {
        let vec = Vec3::new(1.0, 2.0, 2.0);
        assert_eq!(3.0, vec.len())
    }

    #[test]
    fn test_vec3_normalize() {
        let vec = Vec3::new(10.0, 0.0, 0.0);
        assert_eq!(Vec3::new(1.0, 0.0, 0.0), vec.normalize());
    }

    #[test]
    fn test_vec3_square() {
        let vec = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(14.0, vec.square());
    }

    #[test]
    fn test_vec3_rgb() {
        let vec = Vec3::new(1.0, 0.5, 0.0);
        assert_eq!(image::Rgb::<u8>([255, 127, 0]), vec.rgb());
    }

    #[test]
    fn test_vec3_zeros() {
        assert_eq!(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros());
    }
}
