// *Author*: Podtsepko Igor (@IPodtsepko)
use crate::general::{Light, Material};
use crate::geometry::Vec3;
use crate::shapes::Shape;

/// Structure describing ray data.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Ray {
    pub viewer: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Creates and returns a ray with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `viewer` -- radius vector of the observer (camera);
    /// * `direction` -- ray direction.
    pub fn new(viewer: Vec3, direction: Vec3) -> Self {
        Self { viewer, direction }
    }

    fn point(&self, distance: f64) -> Vec3 {
        self.viewer + distance * self.direction
    }
}

/// A structure that combines all the data necessary
/// to calculate the color of a particular ray.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ViewPoint {
    pub ray: Ray,
    pub point: Vec3,
    pub normal: Vec3,
    pub material: Material,
}

impl ViewPoint {
    /// Creates `ViewPoint` with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `ray` -- the data of the ray from this view point;
    /// * `distance` -- distance to the shape;
    /// * `shape` -- the figure from which the ray was reflected (closest to the viewer).
    pub fn new(ray: Ray, distance: f64, shape: &dyn Shape) -> Self {
        let point = ray.point(distance);
        let normal = shape.normal(&point);
        let material = *shape.material();
        Self {
            ray,
            point,
            normal,
            material,
        }
    }

    /// Calculates the direction of the scattered light ray
    pub fn reflect(&self) -> Vec3 {
        Light::reflect(&self.ray.direction, &self.normal)
    }

    /// Calculates the direction of the reflected light ray.
    pub fn refract(&self) -> Vec3 {
        Light::refract(
            &self.ray.direction,
            self.normal,
            self.material.refractive_index(),
        )
    }

    /// Makes an adjustment to the camera position to simplify calculations.
    ///
    /// # Arguments
    ///
    /// * `ray` -- the direction of the ray that hit the camera;
    ///
    /// # Returns
    ///
    /// Adjusted radius vector of the camera.
    pub fn viewer_adjustment(&self, ray: &Vec3) -> Vec3 {
        self.point + 1e-3 * (*ray * self.normal).signum() * self.normal
    }
}
