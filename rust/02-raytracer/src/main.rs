// *Author*: Podtsepko Igor (@IPodtsepko)
use raytracer::{
    colors, defaults,
    general::{Light, Scene},
    geometry::Vec3,
    intensity, materials,
    shapes::ShapeFactory,
};

fn main() {
    let mut scene = Scene::new(Vec3::zeros(), 0.5, colors::CORNFLOWER_CRAYOLA);
    scene.add_shape(ShapeFactory::create_disk(
        materials::GREEN_PLASTIC,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -3.5, -22.0),
        15.0,
    ));
    scene.add_shape(ShapeFactory::create_sphere(
        Vec3::new(-3.0, 0.0, -16.0),
        2.0,
        materials::IVORY,
    ));
    scene.add_shape(ShapeFactory::create_sphere(
        Vec3::new(-1.0, -1.5, -12.0),
        2.0,
        materials::GLASS,
    ));
    scene.add_shape(ShapeFactory::create_sphere(
        Vec3::new(1.5, -0.5, -18.0),
        3.0,
        materials::RED_RUBBER,
    ));
    scene.add_shape(ShapeFactory::create_sphere(
        Vec3::new(7.0, 5.0, -18.0),
        4.0,
        materials::MIRROR,
    ));

    scene.add_light(Light::new(Vec3::new(-20.0, 20.0, 20.0), intensity::LOW));
    scene.add_light(Light::new(Vec3::new(30.0, 50.0, -25.0), intensity::MEDIUM));
    scene.add_light(Light::new(Vec3::new(30.0, 20.0, 30.0), intensity::HIGH));

    scene
        .render(defaults::WIDTH, defaults::HEIGHT)
        .save_with_format(defaults::FILENAME, defaults::IMAGE_FORMAT)
        .expect("Failed to save image");
}
