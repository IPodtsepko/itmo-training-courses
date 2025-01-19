// *Author*: Podtsepko Igor (@IPodtsepko)
use image::Rgb;
use std::ops;

/// Defines a vector in three-dimensional space.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vec3 {
    coordinates: [f64; 3],
}

impl Vec3 {
    /// Creates and returns a vector with the specified set of coordinates.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            coordinates: [x, y, z],
        }
    }

    /// Creates and returns a vector with zero coordinates.
    pub fn zeros() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Returns the norm of the vector.
    pub fn len(&self) -> f64 {
        let len_square: f64 = self.coordinates.map(|x| x.powi(2)).iter().sum();
        len_square.sqrt()
    }

    /// Returns a normalized vector.
    pub fn normalize(self) -> Vec3 {
        self / self.len()
    }

    /// Returns the scalar square of the current vector.
    pub fn square(self) -> f64 {
        self * self
    }

    /// Translates vector coordinates to RGB.
    ///
    /// # Prerequisites
    ///
    ///  Each coordinate lies in the interval [0, 1]
    pub fn rgb(self) -> Rgb<u8> {
        Rgb(self.coordinates.map(|x| (x * 255.0) as u8))
    }
}

impl ops::Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Self) -> Self::Output {
        let mut coordinates = self.coordinates;
        for (i, coordinate) in coordinates.iter_mut().enumerate() {
            *coordinate += rhs.coordinates[i];
        }
        Self { coordinates }
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut coordinates = self.coordinates;
        for (i, coordinate) in coordinates.iter_mut().enumerate() {
            *coordinate -= rhs.coordinates[i];
        }
        Self { coordinates }
    }
}

impl ops::Mul<Vec3> for Vec3 {
    type Output = f64;

    fn mul(self, rhs: Self) -> f64 {
        let mut result: f64 = 0.0;
        for i in 0..3 {
            result += self.coordinates[i] * rhs.coordinates[i];
        }
        result
    }
}

impl ops::Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            coordinates: self.coordinates.map(|coordinate| coordinate * rhs),
        }
    }
}

impl ops::Mul<i32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: i32) -> Self::Output {
        self * (rhs as f64)
    }
}

impl ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl ops::Mul<Vec3> for i32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * (self as f64)
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        -1 * self
    }
}

impl ops::Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Self::Output {
            coordinates: self.coordinates.map(|x| x / rhs),
        }
    }
}

impl ops::Div<i32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: i32) -> Self::Output {
        self / (rhs as f64)
    }
}
