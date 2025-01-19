// *Author*: Podtsepko Igor (@IPodtsepko)
use crate::general::Material;
use crate::geometry::Vec3;
use crate::utils::Ray;

/// An abstract shape containing the methods required to render the image.
pub trait Shape {
    /// Returns the normal to the surface at the `point` point.
    fn normal(&self, point: &Vec3) -> Vec3;
    /// Returns the surface material.
    fn material(&self) -> &Material;
    /// Calculates the distance between the camera and the shape.
    ///
    /// # Arguments
    ///
    /// * `viewer` -- radius-viewer vector;
    /// * `ray` -- the direction of the ray that came into the camera.
    ///
    /// # Returns
    ///
    /// `None` if the beam was not reflected from the shape,
    /// otherwise -- the distance to the figure -- `Some(distance)`.
    ///
    /// # Example
    ///
    /// For the plane given by the equation y = 0, the ray (0, -1, 0) and the camera
    /// located at the point (1, 2, 3), the result will be `Some(2)`. If the beam
    /// direction was (0, 1, 0), then the result would be `None`.
    fn distance(&self, ray: &Ray) -> Option<f64>;
}

pub type ShapePtr = Box<dyn Shape>;

/// Factory for creating shapes.
pub struct ShapeFactory;

impl ShapeFactory {
    /// Creates and returns a boxed sphere with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `center` -- the center of the sphere;
    /// * `radius` -- the radius of the sphere;
    /// * `material` -- the material of the sphere's surface.
    pub fn create_sphere(center: Vec3, radius: f64, material: Material) -> ShapePtr {
        Box::new(Sphere::new(center, radius, material))
    }

    /// Creates and returns a boxed disk with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `material` -- the material of the disk's surface;
    /// * `normal` -- normal to the disk surface;
    /// * `center` -- the center of the disk;
    /// * `radius` -- the radius of the disk.
    pub fn create_disk(material: Material, normal: Vec3, center: Vec3, radius: f64) -> ShapePtr {
        Box::new(Disk::new(material, normal, center, radius))
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Disk {
    material: Material,
    normal: Vec3,
    center: Vec3,
    radius: f64,
}

impl Disk {
    fn new(material: Material, normal: Vec3, center: Vec3, radius: f64) -> Self {
        Self {
            material,
            normal,
            center,
            radius,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Sphere {
    center: Vec3,
    radius: f64,
    material: Material,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Shape for Sphere {
    fn normal(&self, point: &Vec3) -> Vec3 {
        (*point - self.center).normalize()
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn distance(&self, ray: &Ray) -> Option<f64> {
        let l = self.center - ray.viewer;
        let k = l * ray.direction;
        let d = l.square() - k.powi(2);
        if d > self.radius.powi(2) {
            return None;
        }
        let b = (self.radius.powi(2) - d).sqrt();
        if k >= b {
            return Some(k - b);
        }
        if k >= -b {
            return Some(k + b);
        }
        None
    }
}

impl Shape for Disk {
    fn normal(&self, _point: &Vec3) -> Vec3 {
        self.normal
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn distance(&self, ray: &Ray) -> Option<f64> {
        let distance = -(self.normal * (ray.viewer - self.center)) / (self.normal * ray.direction);
        if distance <= 0.0 {
            return None;
        }
        let point = ray.viewer + distance * ray.direction;
        if (point - self.center).len() > self.radius {
            return None;
        }
        Some(distance)
    }
}
