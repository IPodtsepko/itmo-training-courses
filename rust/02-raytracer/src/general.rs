// *Author*: Podtsepko Igor (@IPodtsepko)
use crate::defaults;
use crate::geometry::Vec3;
use crate::shapes::{Shape, ShapePtr};
use crate::utils::{Ray, ViewPoint};
use image::{ImageBuffer, RgbImage};

/// A structure that combines all the elements for image rendering: a set of shapes, lamps, as well
/// as information about the camera.
pub struct Scene {
    shapes: Vec<ShapePtr>,
    lights: Vec<Light>,
    viewer: Vec3,
    viewing_angle: f64,
    background_color: Vec3,
}

impl Scene {
    /// Creates and returns a new empty scene with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `viewer` -- radius-viewer vector;
    /// * `viewing_angle` -- camera viewing angle;
    /// * `background_color` -- background color.
    pub fn new(viewer: Vec3, viewing_angle: f64, background_color: Vec3) -> Self {
        Self {
            shapes: Vec::new(),
            lights: Vec::new(),
            viewer,
            viewing_angle,
            background_color,
        }
    }

    /// Adds a shape to the scene.
    ///
    /// # Arguments
    ///
    /// * `shape` -- pointer to the shape.
    pub fn add_shape(&mut self, shape: ShapePtr) {
        self.shapes.push(shape)
    }

    /// Adds a light to the scene.
    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light)
    }

    /// Renders and returns an image with the specified width and
    /// height parameters (in pixels) in RGB format.
    pub fn render(&self, width: u32, height: u32) -> RgbImage {
        ImageBuffer::from_fn(width, height, |i, j| {
            let x = i as f64 + 0.5 - width as f64 / 2.0;
            let y = -(j as f64 + 0.5) + height as f64 / 2.0;
            let z = height as f64 / (-2.0 * self.viewing_angle.tan());
            let direction = Vec3::new(x, y, z).normalize();
            self.ray_color(direction).rgb()
        })
    }

    fn find_nearest_figure(&self, ray: &Ray) -> Option<(&dyn Shape, f64)> {
        let mut distance = f64::MAX;
        let mut foreground: Option<&ShapePtr> = None;
        for figure in &self.shapes {
            match figure.distance(ray) {
                Some(distance_to_the_figure) => {
                    if distance_to_the_figure < distance {
                        distance = distance_to_the_figure;
                        foreground = Some(figure)
                    }
                }
                _ => continue,
            };
        }
        foreground.map(|figure| (&**figure, distance))
    }

    fn is_shadow(&self, view_point: &ViewPoint, light: &Light) -> bool {
        let direction = light.ray(view_point);
        let adjusted_point = view_point.viewer_adjustment(&direction);
        match self.find_nearest_figure(&Ray::new(adjusted_point, direction)) {
            Some((_, distance)) => {
                (view_point.ray.viewer + (direction * distance) - adjusted_point).len()
                    < light.distance(view_point)
            }
            None => false,
        }
    }

    fn light_components(&self, view_point: &ViewPoint) -> (f64, f64) {
        let mut summary_diffuse = 0.0;
        let mut summary_specular = 0.0;
        for (diffuse, specular) in self
            .lights
            .iter()
            .filter(|&light| !self.is_shadow(view_point, light))
            .map(|light| (light.diffuse(view_point), light.specular(view_point)))
        {
            summary_diffuse += diffuse;
            summary_specular += specular;
        }
        (summary_diffuse, summary_specular)
    }

    fn ray_color(&self, direction: Vec3) -> Vec3 {
        let start_depth: usize = 0;
        self.ray_color_impl(Ray::new(self.viewer, direction), start_depth)
    }

    fn ray_color_impl(&self, ray: Ray, depth: usize) -> Vec3 {
        if depth > defaults::MAX_DEPTH {
            return self.background_color;
        }
        let foreground = self.find_nearest_figure(&ray);
        if foreground.is_none() {
            return self.background_color;
        }
        let (figure, distance) = foreground.unwrap();

        let view_point = ViewPoint::new(ray, distance, figure);
        let (summary_diffuse, summary_specular) = self.light_components(&view_point);

        let ray_color = |direction| {
            self.ray_color_impl(
                Ray::new(view_point.viewer_adjustment(&direction), direction),
                depth + 1,
            )
        };
        let reflect = ray_color(view_point.reflect());
        let refract = ray_color(view_point.refract());

        view_point
            .material
            .lighted(summary_diffuse, summary_specular, reflect, refract)
    }
}

/// Structure describing the light source.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Light {
    position: Vec3,
    intensity: f64,
}

impl Light {
    /// Creates and returns a light source with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `position` -- coordinates of the point where the light source is located;
    /// * `intensity` -- intensity of the light source.
    pub fn new(position: Vec3, intensity: f64) -> Self {
        Self {
            position,
            intensity,
        }
    }

    fn diffuse(&self, view_point: &ViewPoint) -> f64 {
        self.intensity * f64::max(0_f64, view_point.normal * self.ray(view_point))
    }

    fn specular(&self, view_point: &ViewPoint) -> f64 {
        let inverted = -self.ray(view_point);
        let reflect = f64::max(
            0.0,
            -Self::reflect(&inverted, &view_point.normal) * view_point.ray.direction,
        );
        reflect.powf(view_point.material.reflectivity()) * self.intensity
    }

    fn ray(&self, view_point: &ViewPoint) -> Vec3 {
        (self.position - view_point.point).normalize()
    }

    fn distance(&self, view_point: &ViewPoint) -> f64 {
        (self.position - view_point.point).len()
    }

    /// Calculates the direction of the scattered light ray
    pub fn reflect(direction: &Vec3, normal: &Vec3) -> Vec3 {
        (*direction - *normal * 2.0 * (*direction * *normal)).normalize()
    }

    /// Calculates the direction of the reflected light ray.
    pub fn refract(direction: &Vec3, normal: Vec3, refractive_index: f64) -> Vec3 {
        let cos = -f64::max(-1.0, f64::min(1.0, *direction * normal));
        if cos < 0.0 {
            return Self::refract(direction, -normal, 1.0 / refractive_index);
        }
        let k = 1.0 - (1.0 - cos.powi(2)) / refractive_index.powi(2);
        if k < 0.0 {
            Vec3::zeros()
        } else {
            (*direction / refractive_index + normal * (cos / refractive_index - k.sqrt()))
                .normalize()
        }
    }
}

/// A structure describing the material from which the shapes are made.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Material {
    refractive_index: f64,
    albedo: [f64; 4],
    color: Vec3,
    reflectivity: f64,
}

impl Material {
    /// Creates and returns a material with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `refractive_index` -- refractive index of the material;
    /// * `color` -- color of the material;
    /// * `albedo` -- a list of four elements describing the contribution of various components of
    /// color: diffused, reflected and refracted.
    /// * `reflectivity` -- reflectivity of the material.
    pub const fn new(
        refractive_index: f64,
        color: Vec3,
        albedo: [f64; 4],
        reflectivity: f64,
    ) -> Self {
        Self {
            refractive_index,
            albedo,
            color,
            reflectivity,
        }
    }

    /// Returns refractive index of the material.
    pub fn refractive_index(&self) -> f64 {
        self.refractive_index
    }

    /// Returns reflectivity of the material.
    pub fn reflectivity(&self) -> f64 {
        self.reflectivity
    }

    fn lighted(
        &self,
        diffuse_light_intensity: f64,
        specular_light_intensity: f64,
        reflect_color: Vec3,
        refract_color: Vec3,
    ) -> Vec3 {
        self.color * diffuse_light_intensity * self.albedo[0]
            + Vec3::new(1.0, 1.0, 1.0) * specular_light_intensity * self.albedo[1]
            + reflect_color * self.albedo[2]
            + refract_color * self.albedo[3]
    }
}
